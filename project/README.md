# 本地后训练辅助工具

正式训练、交互推理和权重导出使用 **`llamafactory-cli` + 原生 YAML**。本目录只负责离线数据接入、监督标签审计、显存预检、实验回执、角色评测和对比。

- [工作区结构](docs/architecture.md)
- [安装与操作](docs/operations.md)
- [旧命令迁移表](docs/command-migration.md)
- [本次目录迁移与验收](docs/migration-20261018.md)
- [工作区导航源文件](docs/workspace-README.md)

## 分工

`persona-loom/` 维护采集、证据库、生成、审核和偏好对；训练侧仅消费已导出的快照。原生训练配方在 `configs/training/`，推理与导出配方在 `configs/inference/`。辅助配置引用原生配方，不再定义另一套训练参数。

`src/lf_tools/` 是可选包，安装后提供 `lf-tools`。没有 `lf-tools train`、自动生成或自动发布入口。原来的 `posttrain` 源码与实验记录已归档。

当前模型指针在 `state/current-model.json`，仍为 v9；v8 为回退，v10 保持暂停。默认配置不调用付费服务。
