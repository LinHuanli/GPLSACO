# E3四条件完整Static调优与选择封存

四个E3条件的完整Static调优均正常退出，并通过终态独立审计。共5,280次原生调用、692,060,160 search-tour FE、168,960条合法返回路线，已记账求解失败0。每格完整覆盖160配置的1,280搜索调用及10候选的40固定验证调用；没有缩减配置、验证成员或FE，也没有算法墙钟上限。**这些结果完成E3的Static配置选择，不代表RQ3或正式TEST完成。**

科学执行源为7fbe5c827980fe9423d21ee46f3e3b11d0329611，原生fdd64ce保持不变。完整[机器证据](e3_static_terminal_results.json)及[四策略封存](https://github.com/LinHuanli/GPLSACO/blob/16008d9abfd47653712f2c114aa22e91f329fecc/provenance/e3_static_selections_v1.json)绑定每个运行、完整策略、验证成绩、原始任务/审核SHA与实际硬件。封存内容摘要为446611237828d55afe1d5de82971619dd3b00b6c78f66159c4eaba9e7184e589。

## 按预登记验证选出的规则

| 条件 | MNE | 区域ID | 重启请求规则 | 策略SHA前缀 |
|---|---:|---:|---|---|
| alpha-Hard | 8 | 3 | 周期4批次 | 25d3e484 |
| alpha-Escape | 8 | 3 | 每批概率0.25 | a6d2182d |
| POPMUSIC-Hard | 4 | 3 | 周期4批次 | a20a0135 |
| POPMUSIC-Escape | 4 | 3 | 周期4批次 | a20a0135 |

规则仍受共同的合法替代参考约束；缺少alt时保持原参考。POPMUSIC两格选中相同参数，各自全量搜索、验证和图访问限制身份均单独封存。这里的选择只使用原预登记development搜索/验证池，不读取正式TEST；E2的M00仍须等待主线776配置Static/Rule调优，不能用本表替代。

## 独立审计与实际资源

审计从完整原始返回和任务收据重建图/策略/面板，独立重算路线与宏平均，核对160配置覆盖、固定短名单验证、选择排序、完整FE和所有实际CLI终态。每格42,240条路线全部通过，最大距离重算误差分别为8.17e-14、8.17e-14、9.95e-14、9.59e-14。四格均无提交前外来进程、返回后外来/未知进程样本；这只是边界采样，不是连续占用监控。

| 条件 | 实际host / GPU型号 | 原CLI墙钟秒 | CLI / 审计 |
|---|---|---:|---|
| alpha-Hard | cuda05 / RTX PRO 5000 Blackwell | 9,365.22 | 0 / 0 |
| alpha-Escape | cuda05 / RTX PRO 5000 Blackwell | 10,727.63 | 0 / 0 |
| POPMUSIC-Hard | cuda06 / RTX PRO 5000 Blackwell | 9,162.65 | 0 / 0 |
| POPMUSIC-Escape | cuda21 / RTX 6000 Ada Generation | 8,499.19 | 0 / 0 |

UUID、driver、CPU user/system、RSS和内部构造/LS工作量见机器报告。不同型号时间分列，不由本表构造统一加速比；共同主终点仍是4,096 FE/colony。

## 第20个GP按原队列启动

Static-POPMUSIC-Hard结束后，原句柄47754在原cuda06、UUID GPU-90cb0d88-15f1-389f-1f7b-db68d87a3601上自动接续GP-POPMUSIC-Hard-5519，worker PID2105831，run ID57d03475a6cf210b2cf783320170147dd5891b442be44dd85d973fea175797bf。

该运行已取得111调用、14,548,992 FE的不可变前缀，并由独立脚本重放面板/DEAP、原始任务和路线核验通过。它继续完整128×50和全部预登记验证，没有以启动快照替代终态。E3的20个GP现在全部实际启动；先前cuda10两个资源恢复仍按原UUID等待，已返回评价保持入账，不重做。

## 归档及剩余工作

主项目artifacts/runtimes/7fbe5c827980fe9423d21ee46f3e3b11d0329611/static-terminal-evidence-v1.tar.zst保存四个完整运行目录、原CLI日志/资源、独立审计、四策略封存及第20个GP的不可变启动快照。SHA256为b48f29fbcb22760b547787f54621a4bdb784751f5b8f8f23cc20f9745dac2757，189,566,066 bytes；原源码/二进制和全量图准备仍见该版本及其此前运行档案。

下一步继续20个GP完整训练/验证、终态审计与程序选择封存，再按冻结任务矩阵执行TEST、Hard/Escape效应及标签覆盖诊断。E1完整训练与主Static/Rule调优继续，E2工程代码保持独立；不修改当前活动底座，也不提前放行TEST。
