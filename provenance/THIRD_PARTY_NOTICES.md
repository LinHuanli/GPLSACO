# 外部实现与复用声明

`native_faco` 构建直接引用只读的 `../references/ACO-TSP-Adaptive-Tuning` 源文件。诊断 wrapper 调用原生 `run_mfaco` / `run_faco_apt` 并导出路线，不改变其内层算法。作者为 Rafał Skinderowicz，Copyright (c) 2024 RSkinderowicz，遵循 [MIT 原文](licenses/Adaptive-Tuning-MIT.txt)。该快照中内嵌依赖的文件头许可继续随源文件保留；本仓库不重新发布源包或二进制。

实际源文件 SHA-256/Git blob 见 [sources.lock.json](sources.lock.json)。本地缺少Git元数据，只有列出的关键文件与研究方案commit对应blob一致；不宣称整个目录已核验为该commit。

cuOpt仅作为GPU任务分解的只读设计参考，未复制其内部实现。FocusedACO本地无项目LICENSE，未复制。LKH、Concorde、ACOTSP计划作为独立进程的外部基线，其许可不由本仓库统一改写。具体使用前记录实际源版本、许可与构建证据。

额外的 `FACO-Native-EdgeGuard` 是明确标记的适配版本：项目 `.deps/native-edge-guard` 保存MIT源快照与许可证，仅在初始化3-opt排除无向边多重集合完全不变的移动。补丁见 [native_edge_guard.patch](native_edge_guard.patch)，指纹见 [native_edge_guard.json](native_edge_guard.json)。此版本不替换只读原始快照，不冒称作者未修改版本。

`cpp/src/faco_cpu.cpp` 是上述 MIT 来源的 CPU 操作语义移植，原作者版权与完整许可继续适用。它将 Route、checklist LS、选点与稀疏信息素改写为独立接口；已声明的适配包括全零主候选权重的均匀回退、有限 move-evaluation 上限、同时重置 stored/default 的接口。测试 `tests/cpp/test_native_semantics.cpp` 直接编译只读原生源码作 oracle；没有用自写仿制实现替代源级对照。详细边界见 [语义协议](../docs/planning/03_faco_semantics_and_baselines.md)。

`cuda/faco_operations.cu` 延续该Route/MNE/checklist LS语义并改写为block内协作实现；同样保留原作者版权和MIT许可。它通过已核验CPU层作差异检查，当前仅是显式选点的CUDA操作诊断，没有声称完整搜索已等价。
