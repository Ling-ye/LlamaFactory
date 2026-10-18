# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""CUDA operations run in the pinned training environment, isolated from orchestration."""

import copy
import importlib.metadata
import math
import sys
from pathlib import Path

from .artifacts import digest, read_json, read_rows, tree_hashes, verify, write_json


def tokenizer_bundle(request):
    from llamafactory.data import get_template_and_fix_tokenizer
    from llamafactory.hparams import DataArguments, ModelArguments
    from llamafactory.model import load_tokenizer

    bundle = load_tokenizer(ModelArguments(model_name_or_path=request["base"]["path"]))
    args = DataArguments(
        template=request["model"]["template"],
        cutoff_len=10_000_000,
        train_on_prompt=False,
        enable_thinking=request["model"]["enable_thinking"],
        packing=False,
    )
    template = get_template_and_fix_tokenizer(bundle["tokenizer"], args)
    return bundle, args, template


def encode_sft(record, bundle, args, template):
    from llamafactory.data.processor.supervised import SupervisedDatasetProcessor

    messages = record["messages"]
    system = messages[0]["content"] if messages[0]["role"] == "system" else ""
    body = messages[1:] if messages[0]["role"] == "system" else messages
    local = copy.deepcopy(args)
    local.mask_history = record["supervision"] == "last_assistant"
    encoder = SupervisedDatasetProcessor(template, bundle["tokenizer"], bundle.get("processor"), local)
    ids, labels = encoder._encode_data_example(body[:-1], body[-1:], system, "", [], [], [])
    pairs = list(
        template.encode_multiturn(
            bundle["tokenizer"], body, system, "", local.mask_history and not template.preserve_thinking
        )
    )
    expected_ids, expected_labels = template.mm_plugin.process_token_ids(
        [], [], [], [], [], bundle["tokenizer"], bundle.get("processor")
    )
    for i, (source, target) in enumerate(pairs):
        turn_index = len(pairs) - 1 - i if local.mask_history else i
        source_labels = [-100] * len(source)
        if template.efficient_eos and turn_index != 0:
            source_labels[0] = bundle["tokenizer"].eos_token_id
        supervise = not local.mask_history or i == len(pairs) - 1
        expected_ids += source + target
        expected_labels += source_labels + (target if supervise else [-100] * len(target))
    if template.efficient_eos:
        expected_ids.append(bundle["tokenizer"].eos_token_id)
        expected_labels.append(bundle["tokenizer"].eos_token_id)
    if ids != expected_ids or labels != expected_labels or all(t == -100 for t in labels):
        raise ValueError(f"Actual Factory supervision disagrees with requested targets: {record['id']}")
    return {
        "input_ids": ids,
        "labels": labels,
        "attention_mask": [1] * len(ids),
        "images": None,
        "videos": None,
        "audios": None,
    }


def encode_dpo(record, bundle, args, template):
    from llamafactory.data.processor.pairwise import PairwiseDatasetProcessor

    prompt = record["prompt"]
    system = prompt[0]["content"] if prompt[0]["role"] == "system" else ""
    body = prompt[1:] if prompt[0]["role"] == "system" else prompt
    completions = [{"role": "assistant", "content": record[k]} for k in ("chosen", "rejected")]
    encoder = PairwiseDatasetProcessor(template, bundle["tokenizer"], bundle.get("processor"), args)
    chosen, chosen_labels, rejected, rejected_labels = encoder._encode_data_example(
        body, completions, system, "", [], [], []
    )
    for name, ids, labels in (("chosen", chosen, chosen_labels), ("rejected", rejected, rejected_labels)):
        src, dst = template.encode_oneturn(
            bundle["tokenizer"], body + [{"role": "assistant", "content": record[name]}], system, ""
        )
        if template.efficient_eos:
            dst += [bundle["tokenizer"].eos_token_id]
        src, _ = template.mm_plugin.process_token_ids(
            src, None, [], [], [], bundle["tokenizer"], bundle.get("processor")
        )
        if ids != src + dst or labels != [-100] * len(src) + dst:
            raise ValueError("DPO prompt/target mask mismatch")
    return {
        "chosen_input_ids": chosen,
        "chosen_labels": chosen_labels,
        "chosen_attention_mask": [1] * len(chosen),
        "rejected_input_ids": rejected,
        "rejected_labels": rejected_labels,
        "rejected_attention_mask": [1] * len(rejected),
        "images": None,
        "videos": None,
        "audios": None,
    }


def prepare(request):
    from datasets import Dataset, DatasetDict

    verify(request["dataset"])
    output = Path(request["output"])
    bundle, args, template = tokenizer_bundle(request)
    splits, audits = {}, []
    for record in read_rows(Path(request["dataset"]) / "records.jsonl"):
        if record["split"] == "test":
            continue
        value = (encode_sft if request["config"]["stage"] == "sft" else encode_dpo)(record, bundle, args, template)
        lengths = [len(v) for k, v in value.items() if k.endswith("input_ids")]
        if max(lengths) > request["config"]["cutoff_len"]:
            raise ValueError(f"Overlength {record['id']}: {max(lengths)}; truncation is forbidden")
        splits.setdefault(record["split"], []).append(value)
        audits.append(
            {
                "id": record["id"],
                "split": record["split"],
                "tokens": max(lengths),
                "supervision": record["supervision"],
                "encoded_sha256": digest(value),
            }
        )
    DatasetDict({k: Dataset.from_list(v) for k, v in splits.items()}).save_to_disk(str(output / "tokenized"))
    write_json(
        output / "labels.json",
        {
            "passed": True,
            "no_truncation": True,
            "rows": audits,
            "template_default_system": template.default_system,
            "counts": {k: len(v) for k, v in splits.items()},
        },
    )
    write_json(
        output / "tokenized-manifest.json",
        {"files": tree_hashes(output / "tokenized"), "binding_sha256": request["fingerprint"]},
    )


def run_preflight(request):
    import torch
    from datasets import DatasetDict, concatenate_datasets, load_from_disk
    from llamafactory.train.tuner import run_exp
    from transformers import TrainerCallback

    output = Path(request["output"])
    config = copy.deepcopy(request["config"])
    prepared = load_from_disk(config["tokenized_path"])
    data = concatenate_datasets([prepared[split] for split in ("train", "validation") if split in prepared])
    key = "input_ids" if config["stage"] == "sft" else "chosen_input_ids"
    if config["stage"] == "sft":
        index = max(range(len(data)), key=lambda i: len(data[i][key]))
    else:
        index = max(
            range(len(data)),
            key=lambda i: max(len(data[i]["chosen_input_ids"]), len(data[i]["rejected_input_ids"])),
        )
    DatasetDict({"train": data.select([index])}).save_to_disk(str(output / "preflight-tokenized"))
    config.update(
        tokenized_path=str(output / "preflight-tokenized"),
        max_steps=1,
        output_dir=str(output / "preflight-adapter"),
        save_strategy="no",
        do_eval=False,
        eval_strategy="no",
        eval_dataset=None,
    )
    runtime_model = {}

    class FiniteGuard(TrainerCallback):
        def on_train_begin(self, args, state, control, model=None, **kwargs):
            actual = (
                4
                if getattr(model, "is_loaded_in_4bit", False)
                else (8 if getattr(model, "is_loaded_in_8bit", False) else None)
            )
            runtime_model["actual_quantization_bit"] = actual
            requested = config.get("quantization_bit")
            if requested != actual:
                raise ValueError(f"Quantization mismatch: requested {requested}, actual {actual}")

        def on_log(self, args, state, control, logs=None, **kwargs):
            for key, value in (logs or {}).items():
                if isinstance(value, (float, int)) and not math.isfinite(value):
                    raise FloatingPointError(f"Non-finite {key}")

    torch.cuda.reset_peak_memory_stats()
    run_exp(config, callbacks=[FiniteGuard()])
    state = read_json(Path(config["output_dir"]) / "trainer_state.json")
    if state["global_step"] < 1:
        raise ValueError("Training did not perform an optimizer step")
    result = {
        "optimizer_steps": state["global_step"],
        "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
        "loss": state.get("log_history", [{}])[-1].get("train_loss", 0),
        "device_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        **runtime_model,
    }
    result["within_memory_budget"] = result["peak_memory_allocated_bytes"] <= 0.95 * result["device_memory_bytes"]
    write_json(output / "preflight-result.json", result)
    if not result["within_memory_budget"]:
        raise MemoryError("Peak CUDA allocation exceeds 95% of physical VRAM; use an explicit smaller recipe")
    write_json(
        output / "environment.json",
        {k: importlib.metadata.version(k) for k in ("torch", "transformers", "datasets", "peft", "trl")},
    )


def run_inference(request):
    import torch
    from llamafactory.chat import ChatModel
    from llamafactory.chat.hf_engine import HuggingfaceEngine
    from transformers import set_seed

    from .evaluation import execute_cases

    args = {
        "model_name_or_path": request["base"]["path"],
        "template": request["model"]["template"],
        "enable_thinking": request["model"]["enable_thinking"],
        "infer_backend": "huggingface",
        "infer_dtype": "bfloat16",
        "flash_attn": "sdpa",
    }
    if request.get("adapter"):
        args["adapter_name_or_path"] = request["adapter"]["path"]
    model = ChatModel(args)
    engine = model.engine
    system = request["system"]
    if request["protocol"]["system_mode"] == "empty" and (system or engine.template.default_system):
        raise ValueError("Empty-system evaluation detected an injected default system prompt")

    def generate(messages, seed, system_override=None):
        set_seed(seed)
        decoding = request["decoding"]
        active_system = system if system_override is None else system_override
        kwargs, _ = HuggingfaceEngine._process_args(
            engine.model,
            engine.tokenizer,
            engine.processor,
            engine.template,
            engine.generating_args,
            copy.deepcopy(messages),
            active_system,
            input_kwargs=copy.deepcopy(decoding),
        )
        prompt_ids = kwargs["inputs"][0].tolist()
        # Use these very inputs for generation, rather than recording a second approximate rendering.
        with torch.inference_mode():
            result = engine.model.generate(**kwargs)
        tokens = result[0, len(prompt_ids) :].tolist()
        stop_ids = engine.template.get_stop_token_ids(engine.tokenizer)
        finished = bool(tokens and tokens[-1] in stop_ids)
        return {
            "text": engine.tokenizer.decode(tokens, skip_special_tokens=True),
            "input_ids_sha256": digest(prompt_ids),
            "rendered_prompt": engine.tokenizer.decode(prompt_ids, skip_special_tokens=False),
            "output_tokens": len(tokens),
            "finish_reason": "stop" if finished else "length",
            "system": active_system,
            "template_default_system": engine.template.default_system,
        }

    execute_cases(request, generate)


def reload_check(request):
    from llamafactory.chat import ChatModel

    from .models import identify_adapter

    if request.get("adapter"):
        identify_adapter(request["adapter"]["path"], request["base"])
    args = {
        "model_name_or_path": request["base"]["path"],
        "template": request["model"]["template"],
        "enable_thinking": request["model"]["enable_thinking"],
        "infer_backend": "huggingface",
        "infer_dtype": "bfloat16",
        "flash_attn": "sdpa",
    }
    if request.get("adapter"):
        args["adapter_name_or_path"] = request["adapter"]["path"]
    chat = ChatModel(args)
    response = chat.chat([{"role": "user", "content": "你好"}], system="", max_new_tokens=8, do_sample=False)[0]
    write_json(Path(request["output"]) / "reload-check.json", {"loaded": True, "response": response.response_text})


def main():
    operation, request_path = sys.argv[1:]
    request = read_json(request_path)
    actions = {
        "prepare": prepare,
        "preflight": run_preflight,
        "infer": run_inference,
        "reload": reload_check,
    }
    actions[operation](request)


if __name__ == "__main__":
    main()
