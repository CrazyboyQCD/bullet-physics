# SoftBody 移植设计（经典 btSoftBody）

> 设计文档：`bullet-dynamics::softbody` 子模块，移植 `refs/bullet3/src/BulletSoftBody/` 经典体系。
> 用户裁定（2026-08-31）：① 一期做**经典完整体系**（布料/绳索/体积软体 cluster/锚/软体关节/soft-soft），
> `btDeformable*`（NeoHookean/backward-Euler/Krylov）与 `BulletReducedDeformableBody` 记 HANDOFF 待办不实现；
> ② 节点-vs-刚体接触距离用**精确距离**（GJK 点距/解析式）替代 Bullet 的 `btSparseSdf` 体素插值缓存，差异记 deviations。

## 0. Bullet 侧事实（设计依据，已逐项核实源码）

- **步进顺序**（`btSoftRigidDynamicsWorld::internalSingleStepSimulation`）：
  1. `predictUnconstraintMotion`：rigid 速度积分（父类）→ 逐 soft body `predictMotion(dt)`（重力、外力、
     节点积分到预测位 `x`、cluster 更新、bounds、ndbvt/fdbvt 更新、pose 匹配、**清空 rcontacts/scontacts**）
  2. `collisionDetection`：dispatcher 逐 broadphase pair；`(soft, rigid)` → `btSoftRigidCollisionAlgorithm`
     → `defaultCollisionHandler(colObj)` 填 `m_rcontacts`（SDF_RS）或 cluster 关节（CL_RS）；`(soft, soft)` →
     `btSoftSoftCollisionAlgorithm` → `defaultCollisionHandler(psb)` 填 `m_scontacts`（VF_SS）。
     **soft 用预测位 x，rigid 用当前位姿（solve 前）**
  3. islands：soft body 是普通 collision object，参与 union-find（soft 非静态 → 与触碰刚体并岛）
  4. rigid solve（island 内 PGS）→ `integrateTransforms`（rigid 位置积分）
  5. `solveSoftBodiesConstraints`：`solveClusters`（cluster 关节全局迭代）→ 逐 body `solveConstraints()`
     （PBD 位置迭代；接触行内读 rigid **solve 后**速度、施加的冲量**影响 rigid 下一步**——一步滞后耦合，对齐勿"改进"）
  6. self-collision loop：逐 body `defaultCollisionHandler(psb)`（仅 CL_SELF 有效）
  7. `updateSoftBodies` → 逐 body `integrateMotion()`（= `updateNormals`，步末法线刷新）
- **PBD 核心**：`q=x; v+=g·dt; x+=v·dt` → 位置迭代修 `x` → `v=(x−q)·isdt·(1−kDP)`。
- **求解序列默认**（`setSolver(Positions)`，initDefaults 用它）：`psequence=[Anchors, RContacts, SContacts, Linear]`，
  `vsequence/dsequence=[]`；`Velocities` 预设：`vsequence=[Linear(v)]`，`psequence=[Anchors, RContacts, SContacts]`，
  `dsequence=[Linear]`。
- **默认配置**：piterations=1（demo 常设 4）、viterations=0、diterations=0、citerations=4；kDF=0.2、kCHR=1、kKHR=0.1、
  kSHR=1、kAHR=0.7；kVCF=1、kDP/kDG/kLF/kPR/kVC/kMT=0；soft shape **margin=0.25**（接触带语义，巨大）；
  maxDisplacement=1000（防爆炸逐分量钳制）；sleepingThreshold=0.04（但管线不调用 soft 休眠——见 §7）。
- **island 语义**：`updateActivationState` 只遍历 `m_nonStaticRigidBodies`，soft body 永远 ACTIVE_TAG →
  **软体把触碰的刚体岛钉醒，自身不休眠**（Bullet 实际行为）。
- **concave 路径**：soft 整体 AABB 变换到 mesh 局部 → mesh BVH 回调逐三角形 → 每三角形 extruded hull
  （`BT_SOFTBODY_TRIANGLE_EXTRUSION=0.04`）→ 走 soft-vs-convex 算法（SDF 缓存 key=每三角形 hull）。
- **`checkContact`**：`dst = SDF插值(wtr.invXform(x)) − margin`；`dst<0` 成立则接触，
  `cti.offset = −dot(n, x − n·dst)`；PSolve_RContacts 用 `dp=min(dot(x,n)+offset, mrg)`。

## 1. 范围与不做项

**做**：Node/Link/Face/Tetra/Material/Anchor/RContact/SContact/Pose/Cluster 全数据结构；predictMotion/
solveConstraints/staticSolve 求解循环；PSolve_{Links,RContacts,SContacts,Anchors} + VSolve_Links 内核；
applyForces（Aero 7 模型/Pressure/Volume）；K-mean cluster + CL_RS/CL_SS/CL_SELF；LJoint/AJoint/CJoint；
helpers（Rope/Patch/Ellipsoid/FromTriMesh/FromConvexHull + quickhull）；锚（vs rigid）；软体 raycast；
`{translate,rotate,scale,transform,transform_to,set_total_mass,...}` 变换/质量 API；`randomizeConstraints`（MSVC rand 复刻）。

**不做（记 HANDOFF 待办 + deviations）**：`btDeformable*` 新体系（NeoHookean/Corotated/LinearElasticity 力模型、
backward Euler、CG/CR/预条件、deformable 接触投影）；`BulletReducedDeformableBody`；SparseSDF（精确距离替代）；
soft↔multibody 耦合（`SDF_MDF`/`btSoftMultiBodyDynamicsWorld`——后置 Phase F，复用 mb 接触 jacobian 管线）；
Serialize（btSoftBodyData）；GPU solver 接口；debug draw；Note/RenderNode（渲染网格细化 `interpolateRenderMesh`）；
windVelocity 存 worldInfo（并入 body 字段，同 Bullet 实际语义）。

## 2. 数据结构映射（C++ → Rust）

Bullet 用 `Node*` 指针网格 + 运行时 `Body{Cluster*|btRigidBody*}` 包装；Rust 全部换索引
（Bullet 序列化路径 `pointersToIndices/indicesToPointers` 已证明索引同构可行）：

| C++ | Rust（`softbody/body.rs`） | 要点 |
|---|---|---|
| `Material` | `Material { k_lst, k_ast, k_vst, flags }` | Element::m_tag 不移植 |
| `Node` | `Node { x, q, v, f, n, im, area, leaf, battach, index }` | `m_vn/m_effectiveMass*` 是 deformable 专用不移植；`m_constrained` 同 |
| `Link` | `Link { n: [u32;2], mat: u32, rl, c0, c1, c2, c3, bending }` | c0=(im0+im1)/kLST（除法，源码语义）、c1=rl²、c2=1/(grad²·c0) 运行时 |
| `Face` | `Face { n: [u32;3], mat: u32, normal, ra, leaf }` | m_pcontact/m_n0/m_n1 deformable 专用不移植 |
| `Tetra` | `Tetra { n: [u32;4], mat: u32, rv, leaf, c0:[Vector3;4], c1, c2 }` | `m_Dm_inverse/m_F/P_inv` 是 deformable 力模型字段，经典 kVST 只用 c0/c1/c2 |
| `Anchor` | `Anchor { node, body: usize, local, influence, c0, c1, c2 }` | body=RigidBody 索引 |
| `RContact` | `RContact { cti: Cti, node, c0: Matrix3, c1, c2, c3, c4 }` | mb jacobian 字段后置 Phase F |
| `Cti` | `Cti { body: usize, normal, offset }` | m_impulse 仅 multibody 分支用，后置 |
| `SContact` | `SContact { node, face, weights, normal, margin, friction, cfm: [Real;2] }` | |
| `Pose` | `Pose { bvolume, bframe, volume, pos, wgh, com, rot, scl, aqq }` | |
| `Cluster` | `Cluster { masses, nodes, framerefs, framexform, idmass, imass, locii, invwi, com, vimpulses, dimpulses, lv, av, leaf, ndamping, ldamping, adamping, matching, contains_anchor, collide, index }` | btTransform framexform → (Vector3, Quaternion) |
| `Joint/LJoint/AJoint/CJoint` | `enum SoftJoint { Linear(..), Angular(..), Contact(..) }` + `Prepare/Solve/Terminate` 按 enum 分发 | Bullet 虚函数三态 |
| `Body{soft,rigid}` | `enum Side { Cluster { soft, cluster }, Rigid(usize) }` | 冲量分发 helper（§6 关节） |
| `Config` | `Config { ...同名字段, collisions: SoftCollision, psequence: Vec<PSolverKind>, ... }` | fCollision/fMaterial 用 bitflags（HANDOFF §9 约定） |
| `SolverState` | `SolverState { sdt, isdt, velmrg, radmrg, updmrg }` | |
| `btSoftBody` | `SoftBody { cfg, sst, pose, nodes, links, faces, tetras, anchors, rcontacts, scontacts, joints, materials, clusters, ndbvt, fdbvt, cdbvt, bounds, b_update_rt_cst, timeacc, max_speed_squared, ... }` | world 持 `Vec<SoftBody>`；`SoftBodyWorldInfo` 不移植（gravity/air 并入 world 与 Config） |
| `Impulse{velocity,drift}` | `struct Impulse { velocity, drift, as_velocity, as_drift }` | cluster 用 |

**dbvt**：`m_ndbvt`（节点树）/`m_fdbvt`（面树）/`m_cdbvt`（cluster 树）是纯 `btDbvt`（无 pair 缓存）。
从 `broadphase/dbvt.rs` 抽出内部 `Tree` 为 `bullet-collision::dbvt::DbvtTree`（payload 泛型 u32/usize），
补充 `collide_tt`（树-树，VF_SS/CL_SS 用）与叶子 `update`（已有 `update_fat` 语义 = Bullet `update` 的
fat-volume 路径）。**重构触碰 broadphase 热路径，验收=parity 6 场景逐位 + dbvt 全部既有单测**。

## 3. 模块布局

```
crates/bullet-dynamics/src/softbody/
├── mod.rs        // SoftBody 聚合结构 + 对外 API（addForce/setMass/transform/raycast/...）
├── body.rs       // Node/Link/Face/Tetra/Anchor/Cluster/Pose/Material + append* + 质量/惯量设置
├── config.rs     // Config/SolverState/PSolverKind/VSolverKind + SoftCollision bitflags + initDefaults
├── solver.rs     // predict_motion/solve_constraints/static_solve + PSolve_*/VSolve_* 内核
├── forces.rs     // apply_forces（aero node/face + pressure + volume）
├── collide.rs    // check_contact（精确距离）/ SDF_RS / VF_SS / CL_RS / CL_SS / CL_SELF
├── cluster.rs    // generate_clusters（K-mean）/ initialize/update/solve/apply/damp clusters
├── joints.rs     // SoftJoint 三态 Prepare/Solve/Terminate + Side 冲量分发
├── helpers.rs    // create_rope/create_patch/create_ellipsoid/from_tri_mesh/from_convex_hull
├── quickhull.rs  // 最小 quickhull（CreateEllipsoid/FromConvexHull 的面生成）
├── rng.rs        // MSVC rand() 逐位复刻（randomizeConstraints/patch perturbation 用）
└── tests.rs      // 单测（解析对照）
```

facade `bullet/src/lib.rs` re-export；`PhysicsWorld` 增 `soft_bodies: Vec<SoftBody>` +
`add_soft_body/soft_body(_mut)/remove_soft_body`（proxy 级联，同 body removal 模式）。

## 4. world 步进集成（阶段插入点）

对齐 §0 顺序，插入我们 `step_simulation` 的 8 阶段流水线：

| 新阶段 | 位置 | 内容 | Bullet 对应 |
|---|---|---|---|
| 1s | stage 1 后 | `soft_predict_motion(dt)`：重力/外力/节点积分（maxDisplacement 钳制）/clusters/bounds/ndbvt/fdbvt 更新/pose 匹配/清 contacts | `predictMotion` |
| 3s | stage 3 内 | soft proxy AABB ← `m_bounds` | `updateSingleAabb`（soft 作为 collision object） |
| 4b | islands | soft body = 1 个岛节点（与触碰刚体/锚定刚体 union）；soft 恒醒 → 钉醒全岛 | `findUnions` + soft 恒 ACTIVE |
| 5s | stage 5 内分流 | broadphase pair 含 soft owner → 跳过常规窄相，走 soft 碰撞（§5） | dispatcher soft algorithms |
| 7s | stage 7 后（8 前） | `soft_solve_constraints()`：solveClusters（全局）→ 逐 body PBD（接触行冲量此刻写 rigid——已 solve 已积分，影响下一步）→ CL_SELF loop | `solveSoftBodiesConstraints` + self loop |
| 8s | stage 8 旁 | `update_normals()`（=integrateMotion） | `updateSoftBodies` |

注意：锚定刚体（Anchor）在 stage 1s 前须已被唤醒（对齐 Bullet `solveConstraints` 内 `a.m_body->activate()`，
我们在 PSolve_Anchors 装配时 `activate`——需要 world 可变借用，见 §6 借用设计）。

## 5. 碰撞设计（精确距离路线）

`check_contact(body_idx, node_world_x, margin) -> Option<Cti>`：
- **convex 刚体**（Sphere/Box/Capsule/Cylinder/Cone/ConvexHull/Compound→递归子形）：点-vs-凸体 GJK 距离
  （复用 `gjk.rs` 距离行走——零形状侧=点；Sphere 解析 fast path）。`dst = 带符号距离 − margin`，
  `dst<0` → 接触；`normal` = 表面→点 方向（世界系，乘刚体旋转）。
- **Plane**：解析 `dot(n,x)+c`，双侧同 Bullet（SDF 是双代的）。
- **静态 TriangleMesh/Heightfield**：节点附近 AABB → BVH 候选三角形 → `ClosestPtPointTriangle`（CCD 路径已有）
  取最近点距离/法线。**偏差**：Bullet 是 extruded-hull+SDF 插值，我们是精确点-三角形距离 + margin 带
  （deviations 预记 §11-①②）。
- `cti.offset = −dot(n, x − n·dst)`（照抄，PSolve 里 `dp=min(dot(x,n)+offset, mrg)`）。

**SDF_RS**（`(soft, rigid)` pair）：`ndbvt.collide_tv(root, rigid_world_aabb + margin)` → 逐命中节点
`check_contact` → `RContact`：`ra = x − COM_rigid`，`va = rigid.velocity_at_local_point(ra)·sdt`，
`vb = x − q`，`vr = vb − va`，`dn = dot(vr,n)`，`fv = vr − n·dn`，
`fc = kDF·friction_rigid`，`c3 = |fv|² < (dn·fc)² ? 0 : 1−fc`，`c4 = 静态? kKHR : kCHR`，
`c0 = ImpulseMatrix(sdt, ima, imb, invI_rigid, ra)`，`c2 = ima·sdt`。锚定节点（battach）跳过；
动态体 margin 加 `timemargin`（位置步进量，照抄 `dynmargin/stamargin`）。
`(soft, soft)` pair → **VF_SS**：`collide_tt`(ndbvt_a, fdbvt_b) 双向 → 逐 (node, face) 点-三角形含 margin
穿透与重心权重 → `SContact`（cfm 按 `kSHR` 公式，实现时对照 `CollideVF_SS::Process`）。
**CL_RS/CL_SS/CL_SELF**（cluster 模式，Phase D）：cluster 凸包（quickhull 于 cluster 节点，构造临时
`ConvexHull` 支撑走 GJK Band 距离）→ `CJoint`（massmatrix/erp/split 照抄 `ClusterBase::SolveContact`）；
CL_SS 带 cluster 连通性矩阵跳过同体相邻 cluster。

## 6. 求解内核（公式清单，实现逐项对照 btSoftBody.cpp 行号）

| 内核 | Bullet 行号 | 公式/要点 |
|---|---|---|
| predict_motion | 2099 | `sdt=dt·timescale; velmrg=sdt·3; radmrg=margin; updmrg=radmrg/4`；`v += g·sdt` 后 `f·im·sdt`（逐分量钳 ±maxDisp/sdt）；`x += v·sdt`；`f=0`；ndbvt 叶=FromCR(x, radmrg)+速度外推，fdbvt 叶=VolumeOf(face)+pad |
| solve_constraints | 2208 | 装配（links c3=qb−qa、c2=1/(c3²·c0)；anchors c0/c1/c2）→ viterations 序列（后 `x=q+v·sdt`）→ piterations（`ti=isolve/piters` 序列）→ `v=(x−q)·isdt·(1−kDP)` → diterations（q=x，序列，`v += (x−q)·kVCF·isdt`）→ dampClusters+applyClusters(true) |
| PSolve_Links | 4004 | `k=(c1−len)/(c0·(c1+len))·kst`；`xa−=del·k·ima; xb+=del·k·imb`（len=\|del\|²） |
| VSolve_Links | 4027 | `j=−dot(c3, v0−v1)·c2·kst` |
| PSolve_Anchors | 3873 | `vr=(va−vb)+(wa−x)·kAHR`；`impulse=c0·vr·influence`；`x+=impulse·c2`；`rigid.apply_impulse(−impulse, c1)` |
| PSolve_RContacts | 3894 | `vr=vb−va; dn=dot(vr,n)`；`dn≤EPS` 时：`dp=min(dot(x,n)+offset, mrg)`；`impulse=c0·((vr−fv·c3+n·dp·c4)·kst)`；`x−=impulse·c2`；`rigid.apply_impulse(impulse, c1)`（**注意符号：与锚相反，源码即如此**） |
| PSolve_SContacts | 3969 | `vr=(x−q)_node−(p−q)_face`；`dot(vr,nr)<0` 时 `j=margin−(dot(nr,x)−dot(nr,p))`，`corr+=n·j`；`corr−=ProjectOnPlane(vr,nr)·friction`；node `x+=corr·cfm0`，face 三点 `x−=corr·cfm1·w_i` |
| apply_forces | 3750 | pressure：`f += n·(area·(1/\|volume\|·kPR))`；volume：`f += n·(area·(pose.volume−volume)·kVC)`；aero 见 forces.rs |
| ImpulseMatrix×3 | internals 1004-1032 | `Diag(1/dt)·(Diag(ima)+MassMatrix(imb,iwi,r))⁻¹` 等，MassMatrix=iwi−iwi·r×·iwi·r×ᵀ（照抄） |
| clusters | cluster.rs | K-mean（generateClusters k/maxiter）、idmass/imass/locii（点质量惯量）、solveClusters（citerations·SOR=1）、applyClusters(drift)、dampClusters(ndamping/ldamping/adamping)、matching（pose 匹配 kMT） |
| joints L/A/C | joints.rs | LJoint：位置锚定质量矩阵；AJoint：轴对齐（IControl→trait 回调）；CJoint：接触摩擦关节（life/maxlife、split drift）。公式对照 btSoftBody.cpp 3390-3747 |

**借用设计**（world 集成的关键难点）：PSolve_RContacts/Anchors 需要「读 soft 节点 + 读写 rigid 速度」。
方案：`solve_constraints(&mut self.soft_bodies, &mut self.bodies, ...)` 用字段级拆分借用
（soft_bodies 与 bodies 是 PhysicsWorld 的不同字段，天然可同时 &mut）；rigid 冲量经 `apply_impulse`
语义等价函数直写（绕过 solver delta 池——Bullet 此处也是直写 `btRigidBody::applyImpulse`）。

## 7. 休眠/激活

- soft body **恒醒**（对齐 Bullet：`updateActivationState` 只遍历 rigid 列表）；岛合并把触碰刚体钉醒。
- `soft body` 的 `update_deactivation/wants_sleeping/set_zero_velocity` 不接管线（Bullet 同款不调用），deviations 记录。
- max_speed_squared/maxDisplacement 钳制照抄（防爆炸）。

## 8. 确定性（rand 复刻）

`randomizeConstraints` 用 Bullet **内建确定性 LCG**（局部 seed=243703，
`seed = (1664525·seed + 1013904223) & 0xffffffff`，`seed % len` 洗牌 links/faces）——Rust 已逐位复刻，无 rand 依赖。
`CreatePatch` 的 perturbation 用 C `rand()`——probe 两侧均 MSVC 编译，C 期复刻 MSVC rand
（`state = state·214013 + 2531011; (state>>16)&0x7fff`，`srand(0)` 即 state=0）。patch 构建默认
`perturbation=0`。clang 下 Bullet rand 不同——probe 用 MSVC（parity 惯例）。

## 9. quickhull（`quickhull.rs`）

`CreateEllipsoid`（Hammersley 点集，确定性）与 `CreateFromConvexHull`（用户顶点）需要 hull 面生成。
自实现增量式 quickhull（Stan Melax HullLibrary 算法语义的精简版：初始四面体→最远点扩张→共面容差），
输出顶点索引 + 三角面 → links（`idx0<idx1` 条件去重）+ faces。**已知差异**：hull 顶点序/winding 与
Bullet HullLibrary 可能不同 → links 生成顺序不同 → PBD 求解顺序差异 → settle 同类散布（deviations §11-⑤）。

## 10. 分期计划（每期独立提交 + check.py 全绿 + clippy 双 feature 清零）

- **Phase A：骨架 + 绳索动力学** ✅ 完成（2026-08-31）
  body/config/solver(PSolve_Links/VSolve_Links)/helpers(create_rope)；world 阶段 1s/7s 骨架（无碰撞）；
  facade re-export；demo `softbody_rope`（单摆/悬链/自由落体）；单测 8 个：自由落体解析镜像
  （半隐式 Euler 逐节点对照 + f32 消元容差说明）、摆动过底/能量保持、悬链下垂、mass API、
  link 常量公式、randomize 确定性（LCG 复刻）、拉伸回复、world-vs-手动循环逐位一致。
  验收：check.py 全绿（f32/f64 + 10 demo）、clippy 双 feature 清零、parity 6 场景与台账一致（零回归）。
  **实现勘误**：①`randomizeConstraints` 是内建 LCG 非 rand()（§8 已修正，无需 MSVC rand 复刻）；
  ②`updateLinkConstants` 的 `c0 = (im0+im1) / kLST`（除法，§6 表已修正）；
  ③`setTotalMass` 缩放为 `im *= total/mass`。
- **Phase B：SDF_RS 碰撞 + 锚 + broadphase/island 集成** ✅ 完成（2026-08-31 深夜三 + B 后半 probe 同日）
  DbvtTree 从 dbvt.rs 泛化导出（Tree→pub DbvtTree，insert/remove/update_fat/for_each_overlapping/
  optimize_incremental/clear 公开——纯可见性重构，broadphase 逻辑零变）；check_contact（plane 解析/
  凸 GJK 零球点距/静态 mesh·heightfield 点-三角形最近）；SDF_RS→RContact；PSolve_RContacts/PSolve_Anchors
  （solve_constraints 传 &mut bodies 字段级拆分借用）；add_soft_body proxy/ProxyOwner::Soft 分流/
  4b 岛 soft 恒醒钉醒/5s 碰撞/3s bounds 刷 proxy/remove_soft_body + remove_body 锚级联。
  demo：`softbody_rope_ground`（绳落 plane settle 带内 0.149/绳跨球不穿 1.23/双锚 cargo 摆荡保持）。
  **B 后半完成（2026-08-31）**：probe 落地——`tools/softbody_probe.cpp` + `examples/softbody_probe.rs` +
  `tools/build_softbody_probe.py`（链 `BulletSoftBody.lib`，先 cmake build 该 target），场景
  rope_pendulum/rope_ground/cloth_plane ×f32/f64，台账 `docs/bullet-compare.md` "SoftBody probes"。
  读数：纯 PBD（rope_pendulum）打印精度内全对齐（f32 287/300 逐字节）——A/B 求解内核坐实；
  接触场景首触同帧（57）、settle 差 85–86mm（f32=f64 → 结构性）——Bullet 对 plane 走 concave
  三角化+GJK+体素插值的三重近似（`isConvex(plane)`=false，plane 永不直入 GJK 分支），有效接触带
  抬到几何 margin 之上；我们解析距离 settle 在 margin+3mm。deviations §1 已记两行（精确距离量化、
  soft 不休眠）。软体 raycast 挪 C 期（需 fdbvt faces，见下）。
  **实现要点/勘误**：①Bullet 的 `timemargin` 是死代码（wtr 与 ctr 同一对象）→ dynmargin=stamargin；
  ②`dst = 距离 − margin` 语义下 `offset = dst − dot(n,x)`、PSolve 的 dp 即检测时 dst；③**隧穿边界：
  margin 带必须 ≥ 单步位移**（v·dt）——Bullet 0.25 默认即为此设计（15 m/s 上限），测试 margin 0.05+
  7.7 m/s 落速必穿（Bullet 同款物理限制非 bug）；④PBD 链欠收敛拉伸（悬挂 31 节点链 4 迭代 ~60% 伸长）
  与锚摆荡 33% 拉伸是 Bullet 同款特征——断言按形态学特征设；⑤**发现既有 rigid bug（非本轮引入）**：
  宽扁箱（如 1×0.1×1）落 plane 单角接触→自旋→下沉/穿透，HEAD 复现（立方箱正常、parity 场景未覆盖
  扁箱），已记 ignored 测试 + HANDOFF 待办。
  验收：check.py 全绿（f32/f64 + 11 demo）、clippy 双 feature 清零、parity 6 场景逐数字一致、
  dbvt 13 测试绿（mb 路径 diff 为空逻辑论证跳过）。
- **Phase C：Faces + 布料现象** ✅ 完成（2026-08-31 深夜五）
  faces/update_normals/update_area（update_constants 尾部并入，拓扑变化时 fdbvt 重建）；aero 全 7 模型
  （forces.rs，V_*/F_* 双族 + lift/drag + ApplyClampedForce 反转钳制）；pressure/volume（getVolume 散度定理、
  setPose/updatePose——PolarDecompose 延 D 期，kVC 用 setPose(true,false) 即可）；PSolve_SContacts +
  VF_SS（collide_tt 树-树 for_each_overlapping_pair，Bullet push 序；solve_constraints 改关联函数
  `SoftBody::solve_constraints(idx, &mut [SoftBody], bodies)` 字段级拆分——软接触内核跨 body 写双方节点）；
  create_patch（fixeds 全位图/gendiags/perturbation+MsvcRand 复刻——rand 每节点消费 2 次仅 perturbation≠0 时
  影响几何）；软体 raycast（SoftBody::ray_test 全 face 扫描 = Bullet bcountonly 路径 +
  PhysicsWorld::raycast_soft_closest；fdbvt 剪枝记 perf 待办）。demo：`softbody_flag`（第 12 个 check.py demo：
  顶角钉死迎风鼓包 0.745 + 双片缝合枕头 kPR 膨胀 19× 有界）。probe：patch_catenary（ulp 级首差 → 1.6cm
  settle 散布，385 链运算序微差同类）、soft_soft_vf（**首触同帧 43、两片间距两侧都收敛 margin 和 0.504/0.505
  ——VF_SS 带语义对齐**；绝对高差继承下片 plane settle 偏差）。测试 +6（patch 拓扑/面积+闭合盒体积/VF_SS 带内
  保持/raycast 命中+fraction/aero 下风漂移/枕头膨胀）双 feature。
  **实现要点/勘误**：①**fCollision::Default 本树 = SDF_RS 单位**（不含 VF_SS——旧记忆错误已修正），
  soft-soft 需显式 opt-in，且 handler 的 switch 对 `cf & SVSmask` **精确匹配** VF_SS（组合位 fallthrough 无操作，
  照抄）；②Bullet 的 fdbvt 在 predictMotion 头部拓扑变化时 clear+initializeFaceTree（VF_SS 位开才建）——
  追加 faces 不自动建树；③CollideVF_SS 的"跳过含节点面"循环是 Bullet 无效代码（continue 错作用域），
  逐字移植不修；④collideTV/collideTT 的 Bullet 栈序是 **childs[1] 先弹**——DbvtTree::walk_overlapping 已
  对齐（softbody rcontacts 顺序），collideTT 新增同 push 序；⑤CreatePatch 的 rand 消费不设门
  （perturbation 只当系数），perturbation=0 时几何确定 ✓。
- **Phase D：Tetra + Cluster 体积软体** ✅ 完成（2026-08-31 深夜六）
  tetras（`append_tetra`/`set_volume_mass`/`set_volume_density`——**经典系统无 kVST 求解内核**，tetra 只驱动
  质量分布与 `generate_clusters(0)` 的 per-tetra fallback，变形梯度字段是 NeoHookean 专用不移植）；
  `set_total_mass_from_faces`（`setTotalMass(mass, true)` 面积加权——体积软体必需）；
  pose 完整化（`update_pose` 的 PolarDecompose + `scl = aqq·rᵀ·Apq` + maxvolume 钳制——kMT matching 生效）；
  K-mean `generate_clusters`（(i·29873)%k 种子/w=slope 松弛/面 merge/master swap 到 0/fallback 无 master——逐字）+
  cluster 惯量/求解/阻尼/apply；quickhull（自实现：初始单纯形→最远点扩张→horizon 重建，顶点=输入索引升序——
  与 HullLibrary 顶点序差已记 deviations）；CL_RS/CL_SS/CL_SELF（cluster 隐式凸包=节点点集 support，无需 quickhull；
  GJK Band/EPA vs 刚体凸形，plane 走极值顶点解析=精确等价）；CJoint（Prepare/Solve/Terminate 逐字，
  求解拆 `compute`（只读）+ side_apply（逐侧写）两段解借用——关节两侧可在任意 soft body）；
  `generate_bending_constraints`（distance==2 邻接表特化 + Floyd 通用）。
  demo：`softbody_ellipsoid`（第 13 个 check.py demo：体积球落地压扁回弹 min_y 0.18 settle/形态半高 0.95；
  CL_SS 并排推开——**CJoint 摩擦仅 rvac<0 施加是 Bullet 逐字行为**，推开后滑行不减速）。probe：ellipsoid_ground
  （前 21 帧 1e-4 内，settle 差 122mm=Bullet plane 三角化带 vs 解析带——rope_ground 同类簇版；f32=f64 结构性）、
  cluster_pair（**Bullet 自身 CL_SS 大重叠首触数值爆炸穿地 −223m**，我侧稳定推开 settle——gimpact_stack 同类
  "Bullet 自身弱项"，行为分叉记 deviations）。测试 +6（quickhull 3 + ellipsoid 拓扑/体积、面面积/tetra 体积质量、
  bending 计数、cluster master/connectivity、per-face fallback、polar 旋转恢复、pose 旋转跟踪、CL_RS settle、
  CL_SS 分离、hull 内点剔除）双 feature。
  **实现要点/勘误（本轮最重要）**：①**glam 列主序外积陷阱**——Bullet `Apq[row] += a[row]*b` 的行组装照抄进
  glam 会存成**转置**（对称矩阵不暴露，`update_pose` 的 Apq/cluster 协方差拿到逆旋转——pose 测试抓出）；
  正确写法 `m += Matrix3::from_cols(a*b.x, a*b.y, a*b.z)`；②K-mean 空簇会被 Terminate 删（cluster 数 ≤ k+1），
  fallback（k=0）**不装 master**（Bullet 缩进逐字）；③cluster 冲量链：joint→cluster_v_impulse（即时 lv/av +
  vimpulses 累加）→ `solve_constraints` 头 `apply_clusters(false)` 摊到节点位置；④同轴 cluster 堆叠不稳定
  （数值不对称水平滑开——demo 场景 2 改并排推开）。
- **Phase E：软体关节 + API 收口** ✅ 完成（2026-08-31 深夜七）
  `SoftJoint` enum（Linear/Angular/Contact 按 enum 分发 Prepare/Solve/Terminate——Bullet 虚函数三态；
  CJoint 并入后 `solve_clusters` 统一 prepare（mem::replace 借用模式）/compute（只读半冲量）/apply/terminate）；
  LJoint/AJoint 逐字（3587/3630：maxdrift=4/π·16 钳制、ImpulseMatrix/AngularImpulseMatrix、split 双通道）；
  `trait AngularControl`（IControl 的 prepare(joint)/speed(joint,current)，SteerControl 改 refs 的用法保留——
  prepare 用 take 模式解借用；`clone_box` 替代 Bullet 的 static 单例共享）+ `DefaultAngularControl`；
  side 扩展（xform/inv_mass/inv_world_inertia/angular_velocity——cluster 侧 framepos/framerot/invwi/av，
  rigid 侧 world transform/invInertiaTensorWorld/ω）+ side_apply 角向（cluster_va/da_impulse 逐字 1321/1330、
  rigid `apply_torque_impulse` 新增=Bullet applyTorqueImpulse）；append_linear/angular_joint（refs=两侧
  xform 逆变换局部化；world 便捷 `add_soft_linear/angular_joint` 用 master cluster=Bullet 单参重载）；
  变换 API：translate/rotate/scale/transform/transform_to/get_rigid_transform（transform 精确重插 ndbvt
  叶=新 `DbvtTree::update_exact`；updateConstants 重算 rl→缩放后链长跟几何；transform_to 的 Kabsch 旋转
  复用 cluster 极分解 R=Wᵀ（=V·Uᵀ 满秩等价，deviations 记录）；rest_positions=m_X 构造时快照不刷新——
  Bullet quirk 照抄）；cut_link（MatchEdge 无向匹配、link/face 边拆分、!done 回滚节点）；refine
  （ImplicitFn trait、btSymMatrix 对称三角索引、ImplicitSolve 割线迭代 256 步、质量重分配 f 公式、
  face 重处理循环（--i;break）、cut 复制+符号重定向+orphan 清理——`todetach==0` 哨兵 quirk 逐字）。
  demo `softbody_joints`（第 14 个 check.py demo：LJoint 布料拖拽 follow err 0.69/AJoint 马达精确保持
  1.500 rad/s）；测试 +8 双 feature（transform 平移/旋转/缩放、transform_to 拟合恢复、cut_link 拓扑/速度
  继承、refine 插值/质量守恒、cut 符号语义+链清空（Bullet 逐字）、LJoint 锚定抗重力、AJoint 轴对齐收敛、
  马达恒速、torque impulse）。**实现勘误**：①Bullet 的 `m_X` 只在构造函数快照一次（append/cut/refine 不
  刷新）——rest_positions 照抄，append_node 后 transform_to 的拟合对新节点无采样；②refine cut 的节点复制
  条件是 `|eval|<accuracy`（窄带）而 link 重定向条件是 `eval<accuracy`（负半空间）——两个不同判据都逐字；
  ③cluster frame 的 xform 在 rotate/transform 后**延迟一步**（update_clusters 在 predict_motion 里跑）。
- **Phase F（后置待定）**：soft↔mb 耦合（SDF_MDF jacobian 行，复用 `multibody/contact.rs` 管线）。
- perf：C/D 后补 `cloth_wind`/`softbody_rain` perf 场景（双侧 runner 扩展）。

## 11. 偏差预记（deviations 草稿，落地时并入 docs/bullet-deviations.md）

1. **无 SparseSDF**：节点-刚体距离精确化；Bullet 的 0.25 体素 trilinear 插值误差消失 → 接触带内
   settle 位置差异（probe 量化，预期同"margin 壳 44mm 悬浮"量级分类）。
   **已量化（2026-08-31）**：plane 场景 settle 差 85–86mm（Bullet 三重近似把有效带抬到几何 margin 之上，
   我们 settle 在 margin+3mm）——deviations §1 "Soft body: node-vs-rigid distance" 落地。
2. **静态 mesh 不走 extruded hull**：点-三角形精确距离 + margin 带（Bullet 三角形 ±0.04 挤出 + SDF）。
3. **soft 不休眠**（对齐 Bullet 实际管线，非我们简化）。
4. **mb 耦合后置**（Phase F 前软体不与 mb link collider 交互——broadphase pair 过滤）。
5. **quickhull 顶点序差异** → links 顺序 → settle 同类散布。
6. **rand 复刻仅保证 MSVC 工具链 probe 一致**。
7. soft 求解对 rigid 的冲量一步滞后（对齐 Bullet 顺序，勿"改进"为同步）。

## 12. 风险与对策

- **PBD 顺序敏感**：probe 场景优先确定性构建（fixeds/perturbation=0/randomize 关），洗牌场景归混沌类容差。
- **margin 0.25 巨大接触带**：我们的 margin 语义（Bullet `dst<margin` 判接触 + `dp=min(...,mrg)` 钳制）逐式照抄；
  probe rope_ground/cloth_plane 是首个量化点。
- **DbvtTree 重构触碰 broadphase**：Phase B 第一刀，验收=parity 6 场景逐位 + dbvt 13 单测 + mb probes 抽验。
- **borrow 冲突**：world 字段级拆分（soft_bodies/bodies 分开 &mut）；joints 的 `Side` 冲量在
  `PhysicsWorld` 方法内集中分发（不跨对象持有 &mut）。
- **软体混沌本质**：probe 指标用「前 N 帧逐位/高精度 + settle 量（COM/下垂量/节点高度分布/awake）」，
  同 mesh/two_link 分类经验；勿追求长程逐位。
- **数值常量精度相关**（SIMD_EPSILON 分支，f32/f64 双跑）：twist 教训（§6 关节轮）适用——
  `btFuzzyZero/SIMD_EPSILON` 阈值照精度分支。

## 13. 移植对照索引

| 我们 | Bullet |
|---|---|
| softbody/{mod,body,config}.rs | btSoftBody.h + btSoftBody.cpp（结构/append/质量/变换/raycast） |
| softbody/solver.rs | btSoftBody.cpp 2099-2352（predict/solve/static）+ 3873-4072（PSolve/VSolve/getSolver） |
| softbody/forces.rs | btSoftBody.cpp 3750-3815 + addAeroForceToNode/Face |
| softbody/collide.rs | btSoftBodyInternals.h 1440-2000（CollideSDF_RS/VF_SS/CL_*/ClusterBase）+ checkContact 2748 |
| softbody/cluster.rs | btSoftBody.cpp generateClusters/initializeClusters/updateClusters/solveClusters/applyClusters/dampClusters |
| softbody/joints.rs | btSoftBody.cpp 3390-3747（Joint::Prepare/Solve/Terminate 三态）+ Body 515-620 |
| softbody/helpers.rs | btSoftBodyHelpers.cpp 691-1120（Rope/Patch/Ellipsoid/TriMesh/ConvexHull） |
| softbody/quickhull.rs | Extras/ConvexDecomposition/… HullLibrary 语义精简（新实现） |
| world.rs 阶段 1s/3s/4b/5s/7s/8s | btSoftRigidDynamicsWorld.cpp 69-118 + btDiscreteDynamicsWorld 步进 |
