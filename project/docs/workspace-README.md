# LLM 后训练共享资产

本目录位于 `llm-training/llm-training-assets/`，保存模型、运行产物与历史资料，与两个源码仓库并列。

| 位置 | 用途 |
| --- | --- |
| [PersonaLoom](../persona-loom/README.md) | 数据采集、生成、审核和快照 |
| [训练工程](../llama-factory/project/README.md) | 训练配置、辅助工具和评测 |
| models/ | 共享底座及保留模型 |
| artifacts/ | 训练、评测、导出和本地 MLflow |
| archive/ | 历史实验、迁移清单和必要备份 |

当前模型仍为 v9，v8 回退保留，v10 继续暂停。操作见 [指南](../llama-factory/project/docs/operations.md)，布局迁移见 [记录](../llama-factory/project/docs/workspace-assets-20261018.md)。

本导航的版本控制源文件在训练仓库的 `project/docs/workspace-README.md`。
