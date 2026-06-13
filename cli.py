"""
cli.py — Command-line interface for dialact-eval.

Commands:
  dialact-eval ui              — Start the chat UI server
  dialact-eval eval run <path> — Run batch evaluation on a scenario file
  dialact-eval eval list <path>— List scenarios in a file
"""

import asyncio
import os
from pathlib import Path

import click
from dotenv import load_dotenv

load_dotenv()


@click.group()
def main():
    """dialact-eval — voice agent response evaluation toolkit."""
    pass


# =============================================================================
# UI
# =============================================================================

@main.command()
@click.option("--port", "-p", default=8080, show_default=True, help="UI server port")
@click.option("--reload", is_flag=True, help="Enable hot-reload (dev mode)")
def ui(port: int, reload: bool):
    """Start the chat UI server."""
    from core.log import setup_logging
    setup_logging()
    click.echo(f"Starting chat UI on http://localhost:{port}")
    from ui.app import serve
    serve(port=port, reload=reload)


# =============================================================================
# EVAL
# =============================================================================

@main.group()
def eval():
    """Batch evaluation commands."""
    pass


@eval.command("run")
@click.argument("dataset", type=click.Path(exists=True))
@click.option("--output-dir", "-o", default=None, help="Directory for report output")
@click.option("--filter", "-f", "scenario_filter", default=None, help="Only run scenarios matching this substring")
@click.option("--judge", is_flag=True, help="Use LLM-as-judge for goal adherence (requires OPENAI_API_KEY)")
def eval_run(dataset: str, output_dir: str, scenario_filter: str, judge: bool):
    """Run batch evaluation on a scenario YAML file."""
    from core.log import setup_logging
    setup_logging()

    output_dir = output_dir or os.getenv("EVAL_OUTPUT_DIR", "eval/reports")

    from eval.runner import run_eval
    asyncio.run(run_eval(
        dataset_path=dataset,
        output_dir=output_dir,
        use_deepeval_judge=judge,
        scenario_filter=scenario_filter,
    ))


@eval.command("list")
@click.argument("dataset", type=click.Path(exists=True))
def eval_list(dataset: str):
    """List scenarios in a YAML file."""
    from eval.dataset import load_scenario_dataset
    scenarios = load_scenario_dataset(dataset)
    click.echo(f"\n{len(scenarios)} scenarios in {dataset}:\n")
    for s in scenarios:
        answerer = f" ↔ answerer" if s.answerer else ""
        script = f" ({len(s.script)} turns)" if s.script else ""
        click.echo(f"  [{s.difficulty:<6}] {s.id}{answerer}{script}")
        if s.description:
            click.echo(f"           {s.description[:80]}")
    click.echo()
