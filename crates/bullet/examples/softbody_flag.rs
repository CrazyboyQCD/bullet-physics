//! Cloth-phase soft-body demo (phase C of the classic `btSoftBody` port):
//! surface faces, aerodynamic forces and pressure.
//!
//! Two scenes: a flag (patch pinned at its top corners bending in a steady
//! wind through the `V_TwoSided` vertex aero model) and a balloon (a
//! two-sheet pillow with sewn borders inflating under the pressure force).
//!
//! Run with `cargo run --example softbody_flag`.

use bullet::dynamics::{create_patch, patch_fixeds, AeroModel, PhysicsWorld, SoftBody};
use bullet::math::{Real, Vector3};

fn main() {
    let dt: Real = 1.0 / 60.0;

    // Scene 1: flag in a steady wind. The sheet hangs in the x=0 plane
    // (normal +X, facing the wind), pinned at its two top corners.
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
    let flag = world.add_soft_body(create_patch(
        Vector3::new(0.0, 6.0, 0.0),
        Vector3::new(0.0, 6.0, -2.0),
        Vector3::new(0.0, 4.0, 0.0),
        Vector3::new(0.0, 4.0, -2.0),
        9,
        13,
        patch_fixeds::CORNER_00 | patch_fixeds::CORNER_10,
        true,
        0.0,
    ));
    {
        let sb = world.soft_body_mut(flag);
        sb.cfg.piterations = 4;
        sb.cfg.aeromodel = AeroModel::VTwoSided;
        sb.cfg.k_lf = 0.3;
        sb.cfg.k_dg = 0.1;
        sb.wind_velocity = Vector3::new(8.0, 0.0, 0.0);
    }
    let mut bottom_x: Real = 0.0;
    let mut bottom_y: Real = 5.0;
    for _ in 0..600 {
        world.step_simulation(dt);
        let sb = world.soft_body(flag);
        let bottom = sb.nodes[6 * 9 + 4].x; // mid of the lower half
        bottom_x = bottom.x;
        bottom_y = bottom.y;
    }
    println!("flag lower half: x={bottom_x:.3} y={bottom_y:.3}");
    // The sheet bellies downwind like a sail (held by the pinned corners'
    // link tension), the pinned corners never move, and nothing blows up.
    assert!(bottom_x > 0.12, "flag did not catch the wind: {bottom_x}");
    let sb = world.soft_body(flag);
    assert!(
        (sb.nodes[0].x - Vector3::new(0.0, 6.0, 0.0)).length() < 1e-6,
        "pinned corner moved"
    );
    assert!(
        (sb.nodes[8].x - Vector3::new(0.0, 6.0, -2.0)).length() < 1e-6,
        "pinned corner moved"
    );
    assert!(sb.nodes.iter().all(|n| n.x.is_finite()), "flag blew up");

    // Scene 2: balloon — a pillow of two sheets in one soft body, borders
    // sewn with structural links, inflating under the pressure force.
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
    let balloon = world.add_soft_body(build_pillow(9, 5.0, 0.05));
    {
        let sb = world.soft_body_mut(balloon);
        sb.cfg.piterations = 4;
        sb.cfg.k_pr = 2.0;
    }
    let initial_volume = world.soft_body(balloon).get_volume().abs();
    let mut volume = initial_volume;
    let mut max_inflate: Real = 0.0;
    for _ in 0..480 {
        world.step_simulation(dt);
        let sb = world.soft_body(balloon);
        volume = sb.get_volume().abs();
        let com = sb.get_center_of_mass();
        let radius = sb
            .nodes
            .iter()
            .map(|n| (n.x - com).length())
            .fold(0.0, Real::max);
        max_inflate = max_inflate.max(radius);
    }
    println!(
        "balloon volume: {:.4} -> {:.4}, max radius {:.3}",
        initial_volume, volume, max_inflate
    );
    // The pressure force inflates the pillow (volume grows) and the growth
    // stays bounded — the structural links hold the hull together through
    // the PBD-undersconvergence stretch (the same stretch character
    // Bullet's pressurized bodies show), nothing explodes.
    assert!(
        volume > initial_volume * 2.0,
        "balloon did not inflate: {volume} vs {initial_volume}"
    );
    assert!(
        volume < initial_volume * 40.0,
        "balloon over-inflated: {volume} vs {initial_volume}"
    );
    let sb = world.soft_body(balloon);
    assert!(sb.nodes.iter().all(|n| n.x.is_finite()), "balloon blew up");
    println!("OK: flag flies and balloon inflates.");
}

/// A closed two-sheet pillow (`res x res` per sheet, `gap` apart): the
/// upper sheet's faces wind for +Y normals, the lower sheet's for -Y (both
/// outward from the enclosed volume, which the divergence-theorem volume
/// integral needs), and every border node pair is sewn with a structural
/// link.
fn build_pillow(res: usize, y: Real, gap: Real) -> SoftBody {
    let mut positions = Vec::with_capacity(2 * res * res);
    let masses = vec![1.0 as Real; 2 * res * res];
    for iy in 0..res {
        for ix in 0..res {
            let tx = ix as Real / (res - 1) as Real;
            let tz = iy as Real / (res - 1) as Real;
            positions.push(Vector3::new(-1.0 + 2.0 * tx, y, -1.0 + 2.0 * tz));
        }
    }
    for iy in 0..res {
        for ix in 0..res {
            let tx = ix as Real / (res - 1) as Real;
            let tz = iy as Real / (res - 1) as Real;
            positions.push(Vector3::new(-1.0 + 2.0 * tx, y - gap, -1.0 + 2.0 * tz));
        }
    }
    let mut psb = SoftBody::new(&positions, &masses);
    let idx = |sheet: usize, x: usize, y: usize| sheet * res * res + y * res + x;
    for sheet in 0..2 {
        for iy in 0..res {
            for ix in 0..res {
                if ix + 1 < res {
                    psb.append_link(idx(sheet, ix, iy), idx(sheet, ix + 1, iy), None, false);
                }
                if iy + 1 < res {
                    psb.append_link(idx(sheet, ix, iy), idx(sheet, ix, iy + 1), None, false);
                }
                if ix + 1 < res && iy + 1 < res {
                    psb.append_link(
                        idx(sheet, ix, iy),
                        idx(sheet, ix + 1, iy + 1),
                        None,
                        false,
                    );
                    let (n00, n01, n10, n11) = (
                        idx(sheet, ix, iy),
                        idx(sheet, ix + 1, iy),
                        idx(sheet, ix, iy + 1),
                        idx(sheet, ix + 1, iy + 1),
                    );
                    if sheet == 0 {
                        // Upper sheet: outward (+Y) normals.
                        psb.append_face(n00, n01, n11, None);
                        psb.append_face(n11, n10, n00, None);
                    } else {
                        // Lower sheet: outward (-Y) normals.
                        psb.append_face(n00, n10, n11, None);
                        psb.append_face(n11, n01, n00, None);
                    }
                }
            }
        }
    }
    // Sew the borders (the four edges of both sheets share UV layout).
    for i in 0..res {
        psb.append_link(idx(0, i, 0), idx(1, i, 0), None, false);
        psb.append_link(idx(0, i, res - 1), idx(1, i, res - 1), None, false);
        psb.append_link(idx(0, 0, i), idx(1, 0, i), None, false);
        psb.append_link(idx(0, res - 1, i), idx(1, res - 1, i), None, false);
    }
    psb
}
