"""
run_probing.py — Entry point: run the full probing pipeline for all training runs.

Discovers all checkpoint directories, runs the scoring pipeline over every
checkpoint in each run, and saves label_tensor + score_tensor + step_index
for each seed to the results/ directory.

Usage:
    python run_probing.py                          # all runs
    python run_probing.py --seed 42                # single seed
    python run_probing.py --device cpu             # force CPU
    python run_probing.py --batch_size 8           # reduce if OOM

Expected directory structure (after training):
    checkpoints/
        seed42/         ckpt_0000000.pt ... ckpt_0100000.pt
        seed123/        ...
        seed777/        ...
        ablation_6m/    ...
    probe/
        probe_dataset.pt
"""

import argparse
import time
import torch
from pathlib import Path
from typing import List, Optional

from probing import run_probing_pipeline


# ─────────────────────────────────────────────────────────────────────────────
# Run registry — all training runs
# ─────────────────────────────────────────────────────────────────────────────

RUN_REGISTRY = [
    {"seed": 42,   "ckpt_dir": Path("checkpoints/seed42"),       "label": "seed42"},
    {"seed": 123,  "ckpt_dir": Path("checkpoints/seed123"),      "label": "seed123"},
    {"seed": 777,  "ckpt_dir": Path("checkpoints/seed777"),      "label": "seed777"},
    {"seed": 42,   "ckpt_dir": Path("checkpoints/ablation_6m"),  "label": "ablation_6m"},
]


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run probing pipeline over all training run checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Run probing for a single seed only (42, 123, or 777). "
             "Omit to run all seeds.",
    )
    parser.add_argument(
        "--label", type=str, default=None,
        help="Run probing for a specific run label only "
             "(seed42, seed123, seed777, ablation_6m).",
    )
    parser.add_argument(
        "--ckpt_root", type=Path, default=Path("checkpoints"),
        help="Root directory containing per-seed checkpoint subdirectories.",
    )
    parser.add_argument(
        "--probe_path", type=Path, default=Path("probe/probe_dataset.pt"),
        help="Path to the probe_dataset.pt file.",
    )
    parser.add_argument(
        "--results_dir", type=Path, default=Path("results"),
        help="Directory to save probing results (.pt files).",
    )
    parser.add_argument(
        "--ties_log", type=Path, default=Path("results/ties.csv"),
        help="Path to write tie-breaking log.",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device. 'auto' uses CUDA if available.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Inference batch size for attention extraction.",
    )
    parser.add_argument(
        "--use_heldout", action="store_true",
        help="Use heldout probe sequences for scoring (if available).",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Count checkpoints, run single-checkpoint timing test, and exit without saving.",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Filter runs
    runs = RUN_REGISTRY.copy()
    if args.seed is not None:
        runs = [r for r in runs if r["seed"] == args.seed]
    if args.label is not None:
        runs = [r for r in runs if r["label"] == args.label]

    if not runs:
        print(f"[ERROR] No runs matched the given filters.")
        return

    # Validate probe dataset exists
    if not args.probe_path.exists():
        print(f"[ERROR] Probe dataset not found at {args.probe_path}")
        print(f"        Run train_seed42.py first to build the probe dataset.")
        return

    # Dry run mode
    if args.dry_run:
        from data import load_probe_dataset
        from probing import discover_checkpoints, extract_checkpoint, score_all_heads
        from probing.classifier import HeadClassifier, THRESHOLDS
        
        print(f"\n{'=' * 64}")
        print(f"  DRY RUN MODE — Probing Pipeline")
        print(f"{'=' * 64}")
        print(f"  Device       : {device}")
        print(f"  Batch size   : {args.batch_size}")
        print(f"  Probe path   : {args.probe_path}")
        print(f"{'=' * 64}\n")
        
        probe_dict = load_probe_dataset(args.probe_path)
        print(f"  Probe dataset loaded:")
        print(f"    General sequences    : {probe_dict['general_seqs'].shape}")
        print(f"    Induction sequences  : {probe_dict['induction_seqs'].shape}")
        print(f"    Positional sequences : {probe_dict['positional_seqs'].shape}")
        print()
        
        for run in runs:
            label = run["label"]
            ckpt_dir = run["ckpt_dir"]
            
            if not ckpt_dir.exists():
                print(f"  [{label}] Checkpoint directory not found: {ckpt_dir}")
                continue
            
            ckpt_files = discover_checkpoints(ckpt_dir)
            n_ckpts = len(ckpt_files)
            
            print(f"  [{label}] Found {n_ckpts} checkpoints in {ckpt_dir}")
            
            # Time a single checkpoint
            if n_ckpts > 0:
                print(f"  [{label}] Running single-checkpoint timing test...")
                t_start = time.time()
                
                try:
                    extraction = extract_checkpoint(
                        ckpt_path=ckpt_files[0],
                        probe_dict=probe_dict,
                        device=device,
                        batch_size=args.batch_size,
                    )
                    
                    # Create temporary classifier for scoring
                    classifier = HeadClassifier(
                        n_checkpoints=1,
                        n_layers=extraction.config.n_layers,
                        n_heads=extraction.config.n_heads,
                        seed=run["seed"],
                        ties_log_path=Path("/tmp/ties_dryrun.csv"),
                        thresholds=THRESHOLDS,
                    )
                    classifier.register_step(extraction.step)
                    
                    score_all_heads(
                        extraction=extraction,
                        probe_dict=probe_dict,
                        classifier=classifier,
                        ckpt_idx=0,
                        step=extraction.step,
                    )
                    
                    elapsed = time.time() - t_start
                    estimated_total = elapsed * n_ckpts
                    
                    print(f"  [{label}] Single checkpoint time: {elapsed:.1f}s")
                    print(f"  [{label}] Estimated total time: {estimated_total / 60:.1f} min ({estimated_total / 3600:.2f}h)")
                    print(f"  [{label}] Model: {extraction.config}")
                    
                except Exception as e:
                    print(f"  [{label}] ERROR during timing test: {e}")
                    import traceback
                    traceback.print_exc()
            
            print()
        
        print(f"{'=' * 64}")
        print(f"  DRY RUN COMPLETE — No results saved")
        print(f"{'=' * 64}\n")
        return

    # Summary
    print(f"\n{'=' * 64}")
    print(f"  run_probing.py — Probing Pipeline")
    print(f"{'=' * 64}")
    print(f"  Device       : {device}")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  Probe path   : {args.probe_path}")
    print(f"  Results dir  : {args.results_dir}")
    print(f"  Probe set    : {'heldout' if args.use_heldout else 'primary'}")
    print(f"  Runs to probe: {[r['label'] for r in runs]}")
    print(f"{'=' * 64}\n")

    t_total = time.time()
    successful = []
    failed     = []

    for run in runs:
        label    = run["label"]
        seed     = run["seed"]
        ckpt_dir = run["ckpt_dir"]

        suffix = "_heldout" if args.use_heldout else ""
        output_path = args.results_dir / f"results_{label}{suffix}.pt"

        print(f"\n{'─' * 64}")
        print(f"  Processing run: {label}  (seed={seed})")
        print(f"  Checkpoint dir: {ckpt_dir}")
        print(f"  Output path   : {output_path}")
        print(f"{'─' * 64}")

        # Skip if checkpoint directory doesn't exist
        if not ckpt_dir.exists():
            print(f"  [SKIP] Checkpoint directory not found: {ckpt_dir}")
            failed.append(label)
            continue

        # Skip if results already exist
        if output_path.exists():
            print(f"  [SKIP] Results already exist at {output_path}")
            print(f"         Delete the file to re-run probing for this seed.")
            successful.append(label)
            continue

        try:
            run_probing_pipeline(
                ckpt_dir=ckpt_dir,
                probe_path=args.probe_path,
                output_path=output_path,
                ties_log_path=args.ties_log,
                seed=seed,
                device=device,
                batch_size=args.batch_size,
                use_heldout=args.use_heldout,
            )
            successful.append(label)
        except Exception as e:
            print(f"\n  [ERROR] Probing failed for {label}: {e}")
            import traceback
            traceback.print_exc()
            failed.append(label)

    elapsed = time.time() - t_total

    print(f"\n{'=' * 64}")
    print(f"  run_probing.py — Complete")
    print(f"{'=' * 64}")
    print(f"  Total time     : {elapsed / 3600:.2f}h")
    print(f"  Successful     : {successful}")
    print(f"  Failed / skipped: {failed}")
    print(f"\n  Results saved to: {args.results_dir}/")
    for label in successful:
        path = args.results_dir / f"results_{label}.pt"
        if path.exists():
            size_mb = path.stat().st_size / 1e6
            print(f"    results_{label}.pt  ({size_mb:.1f} MB)")
    print(f"{'=' * 64}\n")


if __name__ == "__main__":
    main()
