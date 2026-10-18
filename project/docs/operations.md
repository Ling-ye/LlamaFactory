# 安装与操作

以下在 WSL/Linux 使用。工作区根记为 `llm-training/`；正式原生 CLI 始终从 `llama-factory/` 执行。没有命令会自动启动正式长训练或恢复暂停的 v10。

## 环境

先在源码工作区根执行 `mkdir -p ../llm-training-assets/{models,artifacts,archive}`。源码工作区只保留两个仓库，所有共享资产均放在该同级目录。

```bash
# PersonaLoom：独立运行；无需安装 torch
cd persona-loom
python3 -m venv .venv
.venv/bin/pip install -e './tool[test]'
cd ..

# Factory：按本机 CUDA 版本安装 PyTorch，再安装上游
cd llama-factory
python3 -m venv .venv
.venv/bin/pip install torch==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install -e '.[torch,metrics]'
.venv/bin/pip install transformers==5.8.0
.venv/bin/pip install --no-deps -e ./project

python3 -m venv .venv-tools
.venv-tools/bin/pip install -e './project[test,tracking]'
# OpenCompass 使用独立 .venv-eval；参见 environments/ 中的实际版本记录。
cd ..
```

本次验收使用 Transformers 5.8.0；更新后的上游明确排除了原环境的 5.6.0。实际依赖锁定清单保存在 `project/environments/`。升级环境后重新预检；不要关闭上游版本检查。模型使用官方 `hf download --revision <固定提交>` 下载到共享 `../llm-training-assets/models/`，本工程不复制下载实现。

## 数据快照与监督

```bash
persona-loom/.venv/bin/persona-data export-records \
  --records persona-loom/data/Isekaijoucho/datasets/v10-sft-dpo/sft-v1/records.jsonl \
  --output persona-loom/workspace/exports/my-sft

llama-factory/.venv-tools/bin/lf-tools data import \
  --config llama-factory/project/configs/isekaijoucho-smoke.yaml \
  --output ../llm-training-assets/artifacts/datasets/my-sft
llama-factory/.venv-tools/bin/lf-tools data validate ../llm-training-assets/artifacts/datasets/my-sft
llama-factory/.venv-tools/bin/lf-tools prepare \
  --config llama-factory/project/configs/isekaijoucho-smoke.yaml \
  --dataset ../llm-training-assets/artifacts/datasets/my-sft --output ../llm-training-assets/artifacts/prepared/my-sft
```

先复制配置并把数据 source 指向自己的快照，把原生训练 `output_dir` 改为独立的 `../../llm-training-assets/artifacts/runs/my-sft/adapter`。默认示例会读取仓库内已导出的数据，不采集、不生成。快照保留来源与审核，导出成功不代表人物质量已验收。

混合 all/last assistant 的数据必须经过 `prepare`；不能把两个监督变体直接拼到通用 messages 训练中。prepare 检查实际 tokenizer 标签，拒绝目标被截断，test 不入训练。可选旧桥接：

```bash
persona-loom/.venv/bin/persona-data prepare-factory \
  --records persona-loom/workspace/exports/my-sft/records.jsonl \
  --model ../llm-training-assets/models/Qwen3.5-4B --output ../llm-training-assets/artifacts/prepared/bridge-only \
  --factory-python llama-factory/.venv/bin/python
```

## 预检、原生训练与回执

```bash
llama-factory/.venv-tools/bin/lf-tools preflight \
  --config llama-factory/project/configs/isekaijoucho-smoke.yaml \
  --prepared ../llm-training-assets/artifacts/prepared/my-sft --output ../llm-training-assets/artifacts/checks/my-sft
llama-factory/.venv-tools/bin/lf-tools receipt start \
  --prepared ../llm-training-assets/artifacts/prepared/my-sft --preflight ../llm-training-assets/artifacts/checks/my-sft
cd llama-factory
flock ../../llm-training-assets/artifacts/gpu.lock .venv/bin/llamafactory-cli train ../../llm-training-assets/artifacts/prepared/my-sft/train.yaml
cd ..
llama-factory/.venv-tools/bin/lf-tools receipt finish --run ../llm-training-assets/artifacts/runs/my-sft
```

预检执行一个完整优化器步，检查显存与有限损失，不会静默改成 QLoRA。BF16 与 QLoRA 使用独立配方。所有本地 GPU 命令共用 `../llm-training-assets/artifacts/gpu.lock`。

断点恢复前对未完成的 run 执行 `lf-tools receipt check-resume --run ... --checkpoint ...`，然后官方 CLI 加 `resume_from_checkpoint=...`。已完成的 run 不可覆盖；改数据、底座、参考模型或环境必须新建实验。

DPO 的 `adapter_name_or_path` 与 `ref_model_adapters` 必须指向同一选定 SFT，`ref_model` 必须是同一底座；量化配置也保持一致。偏好数据在 PersonaLoom 构建与审核，训练侧拒绝与父适配器哈希不符的数据。

## 评测、推理与导出

```bash
llama-factory/.venv-tools/bin/lf-tools benchmark freeze \
  --config llama-factory/project/configs/isekaijoucho-smoke.yaml \
  --dataset ../llm-training-assets/artifacts/datasets/my-sft --output ../llm-training-assets/artifacts/protocols/my-smoke
llama-factory/.venv-tools/bin/lf-tools eval \
  --config llama-factory/project/configs/isekaijoucho-smoke.yaml \
  --protocol ../llm-training-assets/artifacts/protocols/my-smoke --adapter ../llm-training-assets/artifacts/runs/my-sft/adapter \
  --output ../llm-training-assets/artifacts/evals/my-smoke

cd llama-factory
flock ../../llm-training-assets/artifacts/gpu.lock .venv/bin/llamafactory-cli chat project/configs/inference/current.yaml
flock ../../llm-training-assets/artifacts/gpu.lock .venv/bin/llamafactory-cli export project/configs/inference/layout-export.yaml
cd ..
```

导出配方示例绑定本次烟测适配器；使用前换成目标适配器与新输出目录。再运行 `lf-tools verify-load --config ... --model ... --output ...` 验证真实加载。适配器分发应同时保存底座哈希、训练 YAML、数据版本和推理 YAML。

评测先冻结协议，保留原始回答和渲染输入；默认无角色 system。完整角色卡对照使用 `isekaijoucho-full-card-smoke.yaml`。使用 `lf-tools compare --left ... --right ... --output ...` 生成匿名材料；若比较不同 system 条件需显式 `--allow-system-difference`。评分和人工标签通过 `review` 导入，评委通过 `calibrate` 校准。进入训练的失败场景会被标记退役，不能重新作为封存验收题。

`lf-tools opencompass --config ... --benchmark project/configs/opencompass/... --output ...` 在隔离环境运行回归。MLflow 新记录写入 `../llm-training-assets/artifacts/tracking/mlflow.sqlite`；历史数据库原封保存在 `../llm-training-assets/artifacts/legacy-posttrain/tracking/`，历史路径查迁移映射，不混写到新实验数据库。

## 测试

```bash
persona-loom/.venv/bin/python -m pytest persona-loom/tool/tests
llama-factory/.venv-tools/bin/python -m pytest llama-factory/project/tests
```

历史评分的两个实际材料 canary 只适用于冻结的 v8 工作区，设置 `PERSONA_HISTORICAL_SCORING_ROOT` 才会运行。当前 v9 工作区不是有效的 v8 夹具；校验发现来源已变化仍会失败，不修改历史来源让它通过。
