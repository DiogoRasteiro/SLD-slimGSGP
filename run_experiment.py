"""
Per-task binary classification experiments with fixed subject-level CV folds.

Prints aggregates only (fold balance, metrics); subject-level outputs stay in
results/, which is gitignored.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from src.core.schema import (
    CV_OUTER_FOLDS,
    DEFAULT_EXPERIMENT_COHORT,
    EXPERIMENT_MODELS,
    EXPERIMENT_SEEDS,
    MODEL_SPECS,
    RESULTS_DIR_NAME,
    TIMING_DIR_NAME,
    TIMING_N_ITER,
)
from src.experiment.cv import run_cv, summarize
from src.experiment.dataset import load_binary_task
from src.experiment.folds import build_folds, default_folds_path, fold_balance, load_folds, save_folds

console = Console()
RESULTS_DIR = Path(__file__).resolve().parent / RESULTS_DIR_NAME


def _table(df: pd.DataFrame, title: str) -> Table:
    table = Table(title=title)
    for col in df.columns:
        table.add_column(str(col), justify="right")
    for row in df.itertuples(index=False):
        table.add_row(*(f"{v:.3f}" if isinstance(v, float) else str(v) for v in row))
    return table


def _cleaning_table(cohort, tasks, data_dir):
    rows = []
    for task in tasks:
        _, r = load_binary_task(cohort, task, data_dir=data_dir)
        rows.append(dict(task=task, raw=r.raw_rows, absent=r.absent_rows, all_zero=r.all_zero_rows,
                         kept=r.kept_rows, positives=r.positive_rows, positive_rate=r.positive_rate))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cohort", "-c", default=DEFAULT_EXPERIMENT_COHORT)
    parser.add_argument("--task", "-t", type=int, nargs="+", required=True,
                        help="Task IDs; the folds are built for (and only valid with) exactly this set.")
    parser.add_argument("--models", "-m", nargs="+", default=list(EXPERIMENT_MODELS), choices=EXPERIMENT_MODELS)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(EXPERIMENT_SEEDS))
    parser.add_argument("--folds", type=int, nargs="+", default=None, help="Subset of outer folds to run.")
    parser.add_argument("--n-folds", type=int, default=CV_OUTER_FOLDS)
    parser.add_argument("--build-folds", action="store_true", help="Build (or rebuild) and save the fold file, then stop.")
    parser.add_argument("--check-folds", action="store_true",
                        help="Recompute and print per-fold positive rates from the saved fold file, then stop.")
    parser.add_argument("--timing", action="store_true",
                        help=f"Time one fold and one seed per model on the first task, with {TIMING_N_ITER} generations "
                             "for models that have generations; writes to results/timing/.")
    parser.add_argument("--run-name", default=None, help="Output folder under results/ (default: derived from cohort, tasks, fold count and fold seed).")
    parser.add_argument("--data-dir", "-d", default=None)
    args = parser.parse_args()

    tasks = sorted(set(args.task))
    folds_path = default_folds_path(RESULTS_DIR, args.cohort, tasks, args.n_folds)

    try:
        if args.build_folds:
            console.print(_table(_cleaning_table(args.cohort, tasks, args.data_dir), "Rows kept per task"))
            assignment = build_folds(args.cohort, tasks, n_folds=args.n_folds, data_dir=args.data_dir)
            save_folds(assignment, folds_path)
            console.print(f"[green]Saved[/green] {folds_path.relative_to(RESULTS_DIR.parent)}")

        if not folds_path.is_file():
            raise FileNotFoundError(f"No fold file for these tasks. Run with --build-folds first ({folds_path.name}).")
        # Re-read from disk even right after building, so the check covers the saved file.
        assignment = load_folds(folds_path)
        assignment.check_matches(args.cohort, tasks)

        if args.build_folds or args.check_folds:
            console.print(_table(fold_balance(assignment, args.data_dir),
                                 f"Fold balance, recomputed from {folds_path.name} (fold seed {assignment.seed})"))
            return

        if args.timing:
            # A new folder each time: resuming would print the previous timings as if new.
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            out = RESULTS_DIR / TIMING_DIR_NAME / f"{assignment.run_stem}_task{tasks[0]}_{stamp}"
            short = {m: {"fixed": {"n_iter": TIMING_N_ITER}} for m in args.models
                     if "n_iter" in MODEL_SPECS[m]["fixed"]}
            metrics = run_cv(assignment, tasks[:1], args.models, args.seeds[:1], out, folds=[0],
                             spec_overrides=short, data_dir=args.data_dir,
                             on_result=lambda r: console.print(f"{r['model']}: {r['elapsed_seconds']:.1f}s"))
            console.print(_table(metrics[["task", "model", "fold", "seed", "elapsed_seconds", "balanced_accuracy"]],
                                 f"Timing (one fold, one seed, {TIMING_N_ITER} generations)"))
            return

        run_name = args.run_name or assignment.run_stem
        out = RESULTS_DIR / run_name
        metrics = run_cv(
            assignment, tasks, args.models, args.seeds, out, folds=args.folds, data_dir=args.data_dir,
            on_result=lambda r: console.print(
                f"task {r['task']} fold {r['fold']} {r['model']} seed {r['seed']}: "
                f"BA {r['balanced_accuracy']:.3f} ({r['elapsed_seconds']:.1f}s)"
            ),
        )
        if not metrics.empty:
            console.print(_table(summarize(metrics), f"Summary ({run_name})"))
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
