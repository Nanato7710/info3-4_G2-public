import pandas as pd
import torch


def trainable_root_name(name: str) -> str:
    parts = name.split(".")

    if len(parts) >= 4 and parts[:2] == ["backbone", "features"]:
        return ".".join(parts[:4])

    if len(parts) >= 3 and parts[:2] == ["backbone", "classifier"]:
        return ".".join(parts[:3])

    return ".".join(parts[:3])


def print_dataset_summary(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    genre_cols: list[str],
) -> None:
    print(f"Train size: {len(train_df)}")
    print(f"Validation size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")

    print("Train genre counts:")
    print(train_df[genre_cols].sum())

    print("Validation genre counts:")
    print(val_df[genre_cols].sum())


def check_dataset_overlap(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    train_ids = set(train_df["ID"].astype(str))
    val_ids = set(val_df["ID"].astype(str))
    test_ids = set(test_df["ID"].astype(str))

    train_val_overlap = train_ids & val_ids
    train_test_overlap = train_ids & test_ids
    val_test_overlap = val_ids & test_ids

    print("\n===== Dataset overlap check =====")
    print("Train / Validation overlap:", len(train_val_overlap))
    print("Train / Test overlap:", len(train_test_overlap))
    print("Validation / Test overlap:", len(val_test_overlap))

    if train_val_overlap or train_test_overlap or val_test_overlap:
        raise ValueError("Dataset split overlap was found.")


def print_trainable_parameter_summary(model: torch.nn.Module) -> None:
    total_params = 0
    trainable_params = 0
    trainable_names = []

    for name, param in model.named_parameters():
        count = param.numel()
        total_params += count

        if param.requires_grad:
            trainable_params += count
            trainable_names.append(name)

    ratio = trainable_params / total_params if total_params else 0.0

    print("\n===== Trainable parameter check =====")
    print(f"Total parameters    : {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({ratio:.2%})")
    print("Trainable roots:")

    roots = sorted(
        {trainable_root_name(name) for name in trainable_names}
    )
    for root in roots:
        print(f"  - {root}")


def assert_trainable_parameters(
    model: torch.nn.Module,
    allowed_prefixes: tuple[str, ...],
) -> None:
    unexpected = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad
        and not name.startswith(allowed_prefixes)
    ]

    if unexpected:
        joined = "\n".join(f"  - {name}" for name in unexpected)
        raise ValueError(
            "Unexpected trainable parameters were found:\n"
            f"{joined}"
        )
