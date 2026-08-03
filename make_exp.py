import argparse
import re
import sys
from pathlib import Path
from shutil import copytree


VALID_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


def parse_args():
    parser = argparse.ArgumentParser(description="Create a new experiment from experiments/template.")
    parser.add_argument("--name", help="Full experiment name, e.g. kazusa-resnet50")
    parser.add_argument("--user-name", help="User name used with --exp-name")
    parser.add_argument("--exp-name", help="Experiment name used with --user-name")
    args = parser.parse_args()

    if args.name and (args.user_name or args.exp_name):
        parser.error("Use either --name or both --user-name and --exp-name.")

    if args.name:
        name = args.name
    elif args.user_name and args.exp_name:
        name = f"{args.user_name}-{args.exp_name}"
    else:
        parser.error("Specify --name, or both --user-name and --exp-name.")

    if not VALID_NAME.fullmatch(name):
        parser.error("Experiment name must contain only letters, numbers, underscores, and hyphens.")

    return name


def update_run_train_script(experiment_dir, name):
    run_train = experiment_dir / "run_train.sbatch"
    if not run_train.exists():
        return

    text = run_train.read_text()
    text = text.replace("#SBATCH --job-name hogehoge", f"#SBATCH --job-name {name}")
    text = text.replace(
        'experiment_dir="experiments/template"',
        f'experiment_dir="experiments/{name}"',
    )
    run_train.write_text(text)


def main():
    name = parse_args()
    project_dir = Path(__file__).resolve().parent
    template_dir = project_dir / "experiments" / "template"
    experiment_dir = project_dir / "experiments" / name
    relative_experiment_dir = Path("experiments") / name

    if experiment_dir.exists():
        print(f"Error: Directory {experiment_dir} already exists. Please choose a different experiment name.", file=sys.stderr)
        return 1

    if not template_dir.is_dir():
        print(f"Error: Template directory {template_dir} does not exist.", file=sys.stderr)
        return 1

    copytree(template_dir, experiment_dir)
    (experiment_dir / "outputs").mkdir(exist_ok=True)
    update_run_train_script(experiment_dir, name)

    print(f"Experiment directory created at {experiment_dir}")
    print()
    print("Run commands:")
    print(f"  uv run python {relative_experiment_dir / 'run_exp.py'}")
    print(f"  sbatch {relative_experiment_dir / 'run_train.sbatch'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
