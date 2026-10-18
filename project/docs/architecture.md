# 双工程结构

```text
llm-training/
├── README.md
├── persona-loom/          # 独立 Git：数据工程
│   ├── tool/             # Python 源码、配置、测试、文档
│   ├── data/             # 明确选定的版本化资料与数据快照
│   ├── benchmarks/       # 角色评分规则、冻结协议
│   └── workspace/        # 本地工作区、导出、生成后端 runtime
├── llama-factory/         # 独立 Git：LLaMA-Factory
│   ├── src/llamafactory/  # 上游训练源码
│   └── project/
│       ├── src/lf_tools/ # 可选辅助工具
│       ├── configs/      # 原生 YAML 与辅助评测配置
│       ├── benchmarks/   # 冻结场景
│       ├── tests/
│       ├── environments/ # 本次验证环境版本
│       ├── state/        # 当前模型与评测政策指针
│       └── docs/
├── models/               # 共享模型，不进 Git
├── artifacts/            # 新数据审计、训练、评测、导出、MLflow
└── archive/              # 历史实验、原始报告、迁移清单与备份
```

上层没有第三个 Git 仓库。导航源文件为训练仓库的 `project/docs/workspace-README.md`；复制到工作区根目录供日常使用。

## 代码边界

- `data.py`：只读取导出快照，保存来源、审核元数据和拆分关系；拒绝跨集合重复及不合法监督模式。
- `worker.py`：训练环境中的唯一 Factory 编码实现，检查逐样本 all/last assistant 标签、错误历史屏蔽和禁止截断。不会把封存测试样本放进 tokenized 数据。
- `bridge.py`：PersonaLoom 可选 `prepare-factory` 的兼容入口，复用同一编码器。
- `native.py`：生成原生训练 YAML、预检与记录官方 CLI 产物。DPO 显式绑定选定 SFT 适配器作为训练初值和固定参考。
- `evaluation.py`：冻结协议、多轮自身历史、空 system 实际输入检查、失败记录、置信区间和匿名 A/B 材料。
- `opencompass*.py`、`regression.py`：隔离环境的能力回归与离线评分接入。
- `tracking.py`：本地 MLflow。不可变文件与哈希是实验依据，MLflow 便于检索。

辅助配置相对其 YAML 文件解析；**原生 Factory YAML 的路径相对 `llama-factory/` 工作目录**。统一运行时路径在 `configs/runtime.yaml`。迁移后的环境均有自己的 Python；不能把 venv Python 的符号链接解析为系统 Python。

## 历史与活跃状态

历史 `lab/` 在 `archive/lab/`；旧训练产物在 `artifacts/legacy-saves/` 和 `artifacts/legacy-posttrain/`。历史 JSON、manifest、报告保持原文，旧路径由迁移映射解释。PersonaLoom 的历史评分工具通过 `workspace_paths.py` 解析归档来源；它们仍保留原来的冻结来源检查。

当前歌词政策仍是 `recognizable-lyrics-2.0.1`：任意位置的可辨认歌词、片段、改写、拼接和翻译均可，不能把固定后缀命中率作为新验收门槛。辅助工具的昵称/身份标记是程序诊断；人物感、歌词辨认和自然度需按原政策审核，不能由标记率推定合格。

两个角色配置示例（异世界情绪、原创星灯）和普通问答配置共用代码。新增角色应更换角色包、数据、评测协议和原生训练 YAML。
