import argparse
import sys
from pathlib import Path

OUTPUTS_ROOT = Path(__file__).resolve().parents[1]
if str(OUTPUTS_ROOT) not in sys.path:
    sys.path.insert(0, str(OUTPUTS_ROOT))

from artifact_common import run_training
from criterion import build_criterion
from evaluate import evaluate_model
from model import build_model
from optimizer import build_optimizer
from train import train_one_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one frozen final model seed.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True, choices=(42, 43, 44))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = run_training(
        model_id="final-tri-model",
        config_path=args.config.resolve(),
        seed=args.seed,
        force=args.force,
        smoke=args.smoke,
        build_model=build_model,
        build_criterion=build_criterion,
        build_optimizer=build_optimizer,
        train_one_epoch=train_one_epoch,
        evaluate_model=evaluate_model,
    )
    print(f"completed: {path}")


if __name__ == "__main__":
    main()
