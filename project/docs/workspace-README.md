# 本地 LLM 后训练工作区

这是两个独立源码仓库与共享资产的工作区，上层不建立 Git 仓库。

| 位置 | 用途 |
| --- | --- |
| [persona-loom/](persona-loom/README.md) | 采集、证据、生成、审核、SFT 与 DPO 数据快照 |
| [llama-factory/project/](llama-factory/project/README.md) | 自有配置、数据审计、评测、迁移与操作说明 |
| [llama-factory/](llama-factory/README_zh.md) | 上游训练、聊天、导出；统一使用 llamafactory-cli |
| models/ | 共享底座与保留模型，不进 Git |
| artifacts/ | 新训练、评测、导出、MLflow，以及迁入的旧训练产物 |
| archive/ | 历史实验、原始报告、迁移映射和必要备份 |

当前模型仍为 v9，v8 为回退；v10 保持暂停。活跃指针见 [current-model.json](llama-factory/project/state/current-model.json)。

先读 [操作指南](llama-factory/project/docs/operations.md) 和 [迁移说明](llama-factory/project/docs/migration-20261018.md)。两个远端分别为 [PersonaLoom](https://github.com/Ling-ye/persona-loom) 和 [LLaMA-Factory](https://github.com/Ling-ye/LlamaFactory)。

本导航的版本控制源文件保存在 llama-factory/project/docs/workspace-README.md。
