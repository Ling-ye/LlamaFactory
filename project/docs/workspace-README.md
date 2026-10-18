# LLM 后训练资产

本目录 `llm-training-assets/` 保存模型、运行产物与历史资料。源码工作区 `../llm-training/` 只包含两个独立 Git 仓库。

| 位置 | 用途 |
| --- | --- |
| [PersonaLoom](../llm-training/persona-loom/README.md) | 数据采集、生成、审核与快照 |
| [训练工程](../llm-training/llama-factory/project/README.md) | 训练配置、辅助工具与评测 |
| models/ | 共享底座及保留模型 |
| artifacts/ | 训练、评测、导出和本地 MLflow |
| archive/ | 历史实验、迁移清单与必要备份 |

当前模型仍为 v9，v8 回退保留，v10 继续暂停。操作见 [指南](../llm-training/llama-factory/project/docs/operations.md)，外置记录见 [迁移报告](../llm-training/llama-factory/project/docs/external-assets-20261018.md)。

本导航的版本控制源文件在训练仓库的 `project/docs/workspace-README.md`。
