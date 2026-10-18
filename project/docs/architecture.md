# 双工程与外部资产结构

```text
projects/
├── llm-training/                   # 仅包含两个 Git 仓库
│   ├── persona-loom/               # 数据工程
│   │   ├── tool/                   # 源码、配置、测试、文档
│   │   ├── data/                   # 版本化资料与数据快照
│   │   ├── benchmarks/             # 角色评分规则与冻结协议
│   │   └── workspace/              # 数据工程本地工作区（Git 忽略）
│   └── llama-factory/              # 训练工程
│       ├── src/llamafactory/       # 上游源码
│       └── project/               # 自有配置、辅助工具、测试、状态和文档
└── llm-training-assets/            # 外部资产，不是 Git 仓库
    ├── README.md
    ├── models/                    # 共享模型
    ├── artifacts/                 # 训练、评测、导出与 MLflow
    └── archive/                   # 历史实验、迁移清单与必要备份
```

`llm-training/` 没有上层 Git 仓库，也不再放导航文件或资产符号链接。导航在各仓库 README 和外部资产目录的 README；后者的源文件为 `project/docs/workspace-README.md`。

## 代码边界

- PersonaLoom 负责采集、证据、生成、审核与数据快照导出，训练端只读取离线快照。
- `data.py` 保存来源与审核记录，检查拆分关系和监督模式。
- `worker.py` 是训练环境中的唯一 Factory 编码实现，检查 all/last assistant、错误历史屏蔽与禁止截断；test 不进入训练编码。
- `bridge.py` 是 PersonaLoom 可选 `prepare-factory` 的兼容入口。
- `native.py` 生成原生 YAML、预检并记录官方 CLI 产物。DPO 显式绑定选定 SFT 初值与固定参考。
- `evaluation.py` 维护冻结协议、多轮自身历史、空 system 检查、失败记录与匿名 A/B 材料。
- `opencompass*.py` 接入隔离环境的回归评分；`tracking.py` 记录本地 MLflow。
- `paths.py` 只在读取旧记录路径时解析迁移位置，不改写冻结 manifest 或 JSON 中的原始值。

辅助配置路径相对 YAML 文件；**原生 Factory YAML 路径相对 `llama-factory/` 工作目录**。例如原生模型路径为 `../../llm-training-assets/models/Qwen3.5-4B`。锁和 MLflow 配置统一在 `configs/runtime.yaml`。

## 历史与活跃状态

旧报告与 manifest 保持原文。两次迁移依次将原目录整理到资产目录、再整体移到工作区外，迁移记录保存在外部 `archive/migration/`。读取旧 `llm-training/models`、`artifacts`、`archive` 或更早的 `lab`、`saves` 引用时，兼容层解析到实际位置。

当前模型指针在 `state/current-model.json`，路径仍相对源码工作区根，资产路径以 `../llm-training-assets/` 开始。v9 为当前版本、v8 为回退；v10 仍暂停。当前歌词政策仍为 `recognizable-lyrics-2.0.1`，可辨认的片段、改写、拼接和翻译沿用原规则。

移动到其他磁盘时应先更新原生 YAML、运行时配置和状态指针。兼容层可用 `LLM_TRAINING_ASSETS` 指定旧记录的资产解析根；该变量不会替代原生 CLI YAML 中的显式路径。
