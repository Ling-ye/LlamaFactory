# 2026-10-18 工作区内的共享资产

用户选择将两个 Git 仓库与共享资产放在同一个工作区：

```text
llm-training/
├── persona-loom/
├── llama-factory/
└── llm-training-assets/
```

已将原同级 `../llm-training-assets/` 整体移动到工作区内。资产目录没有 Git 仓库，两个源码仓库的边界和历史保持独立。

本次使用同一文件系统的目录重命名移动 19,598 个文件项、54,694,098,311 bytes 普通文件；移动前后逐项比较大小、mtime、inode 与符号链接目标。603 个保护文件在移动前后分别核对 SHA-256，一致通过。

## 更新内容

- 原生训练、推理、导出 YAML，辅助配置、PersonaLoom tokenizer 路径及当前模型指针改为工作区内的资产目录。
- 读取旧同级路径与更早的工作区根路径时，由兼容层解析到实际位置；历史 JSON、报告和 manifest 保持原文。
- MLflow 当前数据库的 15 项位置记录已更新，修改前保存了独立 SQLite 备份；历史数据库未改写。
- v9 相对符号链接继续指向同一 checkpoint，v8 回退保留，v10 保持暂停。
- 忽略规则继续排除误放入两个 Git 仓库内的资产、权重、环境和缓存。
- 导航移到资产目录 README；工作区根目录只保留上述三个目录。

完整文件清单、保护文件哈希、迁移日志和数据库备份位于 `llm-training-assets/archive/migration/20261018-workspace-assets/`。前两次布局的记录保留，当前以本文和 [工作区结构](architecture.md) 为准。

一次性迁移工具为 `project/scripts/relocate_assets.py`，要求显式传入 workspace、source 与 assets，仅允许两种已知资产位置间的同盘迁移；旧的单向外置脚本已由该工具替代。

## 验证结果

PersonaLoom 587 项通过、12 项跳过；训练辅助工具 69 项通过。已实际验证旧同级路径与原工作区路径读取、冻结产物哈希、v9/v8 适配器与底座匹配、状态指针哈希，以及官方 CLI v9 空提示推理。工作区根目录严格为两个仓库和一个共享资产目录。

[机器可读迁移与验证摘要](migration/20261018-workspace-assets.json)。
