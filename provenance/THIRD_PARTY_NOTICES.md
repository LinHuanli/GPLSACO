# 外部实现与复用声明

`native_faco` 构建直接引用只读的 `../references/ACO-TSP-Adaptive-Tuning` 源文件。诊断 wrapper 调用原生 `run_mfaco` / `run_faco_apt` 并导出路线，不改变其内层算法。作者为 Rafał Skinderowicz，Copyright (c) 2024 RSkinderowicz，遵循 [MIT 原文](licenses/Adaptive-Tuning-MIT.txt)。该快照中内嵌依赖的文件头许可继续随源文件保留；本仓库不重新发布源包或二进制。

实际源文件 SHA-256/Git blob 见 [sources.lock.json](sources.lock.json)。本地缺少Git元数据，只有列出的关键文件与研究方案commit对应blob一致；不宣称整个目录已核验为该commit。

cuOpt仅作为GPU任务分解的只读设计参考，未复制其内部实现。FocusedACO本地无项目LICENSE，未复制。LKH、Concorde、ACOTSP计划作为独立进程的外部基线，其许可不由本仓库统一改写。具体使用前记录实际源版本、许可与构建证据。

额外的 `FACO-Native-EdgeGuard` 是明确标记的适配版本：项目 `.deps/native-edge-guard` 保存MIT源快照与许可证，仅在初始化3-opt排除无向边多重集合完全不变的移动。补丁见 [native_edge_guard.patch](native_edge_guard.patch)，指纹见 [native_edge_guard.json](native_edge_guard.json)。此版本不替换只读原始快照，不冒称作者未修改版本。
