# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""Supporting commands only: official llamafactory-cli owns training and export."""

import json
from pathlib import Path

import typer

from . import data, evaluation, native
from .config import Bundle

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)
datasets = typer.Typer(no_args_is_help=True)
benchmarks = typer.Typer(no_args_is_help=True)
receipts = typer.Typer(no_args_is_help=True)
app.add_typer(datasets, name="data")
app.add_typer(benchmarks, name="benchmark")
app.add_typer(receipts, name="receipt")


def show(value):
    typer.echo(json.dumps(value, ensure_ascii=False, indent=2, default=str))


@datasets.command("import")
def import_data(config: Path = typer.Option(...), output: Path = typer.Option(...)):
    """Import a dataset export; generation belongs to PersonaLoom."""
    show(data.build(Bundle(config), output))


@datasets.command("validate")
def validate(dataset: Path):
    show(data.validate(dataset))


@app.command("prepare")
def prepare(config: Path = typer.Option(...), dataset: Path = typer.Option(...), output: Path = typer.Option(...)):
    """Audit labels and emit native train.yaml; does not start training."""
    show({"prepared": native.prepare(Bundle(config), dataset, output)})


@app.command("preflight")
def preflight(config: Path = typer.Option(...), prepared: Path = typer.Option(...), output: Path = typer.Option(...)):
    """Run one independent optimizer step to verify the explicit memory recipe."""
    show({"preflight": native.preflight(Bundle(config), prepared, output)})


@receipts.command("start")
def start(prepared: Path = typer.Option(...), preflight: Path = typer.Option(...)):
    """Create a receipt immediately before invoking the official CLI."""
    show({"run": native.start_receipt(prepared, preflight)})


@receipts.command("finish")
def finish(run: Path = typer.Option(...)):
    """Verify and track the result after the official CLI exits successfully."""
    show({"run": native.finish_receipt(run)})


@receipts.command("check-resume")
def resume(run: Path = typer.Option(...), checkpoint: Path = typer.Option(...)):
    show(native.check_resume(run, checkpoint))


@benchmarks.command("freeze")
def freeze(
    config: Path = typer.Option(...), output: Path = typer.Option(...), dataset: Path | None = typer.Option(None)
):
    show({"protocol": evaluation.freeze(Bundle(config), output, dataset)})


@app.command("eval")
def evaluate(
    config: Path = typer.Option(...),
    protocol: Path = typer.Option(...),
    output: Path = typer.Option(...),
    adapter: Path | None = typer.Option(None),
    resume: bool = typer.Option(False),
):
    """Audited multi-turn evaluation; everyday inference uses llamafactory-cli chat."""
    show({"evaluation": evaluation.evaluate(Bundle(config), protocol, output, adapter, resume)})


@app.command("compare")
def compare(
    left: Path = typer.Option(...),
    right: Path = typer.Option(...),
    output: Path = typer.Option(...),
    allow_system_difference: bool = typer.Option(False),
):
    show({"comparison": evaluation.compare(left, right, output, allow_system_difference)})


@app.command("review")
def review(comparison: Path = typer.Option(...), reviews: Path = typer.Option(...), output: Path = typer.Option(...)):
    show({"review": evaluation.import_reviews(comparison, reviews, output)})


@app.command("calibrate")
def calibrate(
    model_reviews: Path = typer.Option(...), human_reviews: Path = typer.Option(...), output: Path = typer.Option(...)
):
    from .calibration import calibrate as run

    show({"calibration": run(model_reviews, human_reviews, output)})


@app.command("opencompass")
def opencompass(
    config: Path = typer.Option(...), benchmark: Path = typer.Option(...), output: Path = typer.Option(...)
):
    from .opencompass import run

    show({"evaluation": run(Bundle(config), benchmark, output)})


@app.command("verify-load")
def verify_load(
    config: Path = typer.Option(...),
    output: Path = typer.Option(...),
    model: Path | None = typer.Option(None),
    adapter: Path | None = typer.Option(None),
):
    show({"verification": native.verify_reload(Bundle(config), output, model, adapter)})


@app.command("inspect")
def inspect(config: Path = typer.Option(...)):
    show(Bundle(config).snapshot())


if __name__ == "__main__":
    app()
