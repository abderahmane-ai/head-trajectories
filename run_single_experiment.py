"""Run one full experiment profile end-to-end outside notebooks."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.profiles import get_profile, list_profiles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single Head Trajectories experiment profile end-to-end.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--profile",
        type=str,
        required=False,
        help="Experiment profile name.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Training seed for this run.",
    )
    parser.add_argument(
        "--artifact_root",
        type=Path,
        default=Path("artifacts"),
        help="Root directory for all generated artifacts.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Training/probing device.",
    )
    parser.add_argument(
        "--reset_run",
        action="store_true",
        help="Delete the per-seed artifact directory before running.",
    )
    parser.add_argument(
        "--rebuild_probe",
        action="store_true",
        help="Rebuild the shared probe dataset for this profile.",
    )
    parser.add_argument(
        "--skip_train",
        action="store_true",
        help="Skip training and reuse existing checkpoints.",
    )
    parser.add_argument(
        "--skip_probe",
        action="store_true",
        help="Skip probing and reuse existing results.",
    )
    parser.add_argument(
        "--skip_analysis",
        action="store_true",
        help="Skip analysis/plot generation.",
    )
    parser.add_argument(
        "--list_profiles",
        action="store_true",
        help="List available profiles and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_profiles:
        print("Available profiles:")
        for profile in list_profiles():
            print(f"  - {profile.name}: {profile.description}")
        return

    if not args.profile:
        raise SystemExit("--profile is required unless --list_profiles is used")

    from experiments.runner import run_full_single_experiment

    profile = get_profile(args.profile)
    manifest = run_full_single_experiment(
        profile=profile,
        seed=args.seed,
        artifact_root=args.artifact_root,
        device=args.device,
        reset_run=args.reset_run,
        rebuild_probe=args.rebuild_probe,
        skip_train=args.skip_train,
        skip_probe=args.skip_probe,
        skip_analysis=args.skip_analysis,
    )

    print("\nRun complete.")
    print(f"Profile      : {profile.name}")
    print(f"Seed         : {args.seed}")
    print(f"Artifact root: {args.artifact_root}")
    print(f"Seed dir     : {manifest['artifacts']['seed_dir']}")
    print(f"Manifest     : {Path(manifest['artifacts']['seed_dir']) / 'manifest.json'}")


if __name__ == "__main__":
    main()
