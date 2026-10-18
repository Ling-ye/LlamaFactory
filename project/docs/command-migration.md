# 原命令到新命令

原命令只用于读历史，旧脚本不再是公共入口。

| 原用法 | 新用法 |
| --- | --- |
| posttrain models fetch | 官方 hf download，固定 revision；模型写入共享 models/ |
| posttrain data generate | PersonaLoom 的 collect / build / curriculum；训练端不调用生成 |
| posttrain data build | persona-data export-records / preferences export，然后 lf-tools data import |
| posttrain data validate | lf-tools data validate |
| posttrain train | lf-tools prepare → preflight → receipt start；llamafactory-cli train；receipt finish |
| posttrain train --resume… | lf-tools receipt check-resume；llamafactory-cli train … resume_from_checkpoint=… |
| posttrain eval / compare | lf-tools eval / compare |
| posttrain benchmark freeze | lf-tools benchmark freeze |
| posttrain review / calibrate | lf-tools review / calibrate |
| posttrain opencompass | lf-tools opencompass |
| posttrain export --merge | llamafactory-cli export 原生 YAML；lf-tools verify-load |
| 角色专用聊天脚本 | llamafactory-cli chat project/configs/inference/current.yaml |
| persona-data prepare-factory 在旧训练环境执行 | PersonaLoom 环境中执行，加 --factory-python 指向独立训练 Python |
| persona-data/.venv/bin/python | persona-loom/.venv/bin/python |
| 根目录 .venv/bin/llamafactory-cli | llama-factory/.venv/bin/llamafactory-cli |

保留 `posttrain-binding.json` 作为已有适配器的绑定文件名，避免破坏历史兼容；这不恢复 posttrain 训练入口。
