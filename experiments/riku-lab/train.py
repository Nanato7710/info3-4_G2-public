import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


def feature_distillation_loss(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
) -> torch.Tensor:
    if student_features.shape != teacher_features.shape:
        raise ValueError(
            "Student and teacher feature shapes do not match: "
            f"student={tuple(student_features.shape)}, "
            f"teacher={tuple(teacher_features.shape)}"
        )

    student_features = F.normalize(
        student_features,
        dim=1,
    )
    teacher_features = F.normalize(
        teacher_features,
        dim=1,
    )

    return 1.0 - F.cosine_similarity(
        student_features,
        teacher_features,
        dim=1,
    ).mean()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float = 1.0,
    teacher_model: nn.Module | None = None,
    distill_alpha: float = 0.0,
) -> dict[str, float]:
    model.train()
    if teacher_model is not None:
        teacher_model.eval()

    total_loss = 0.0
    total_genre_loss = 0.0
    total_distill_loss = 0.0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        images = torch.nan_to_num(
            images,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )
        labels = torch.nan_to_num(
            labels,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )
        labels = labels.clamp(0.0, 1.0)

        optimizer.zero_grad(set_to_none=True)

        if teacher_model is not None and distill_alpha > 0:
            logits, student_features = model(
                images,
                return_features=True,
            )

            with torch.no_grad():
                teacher_features = teacher_model(images)
        else:
            logits = model(images)
            student_features = None
            teacher_features = None

        if not torch.isfinite(logits).all():
            print("logits に NaN/Inf が出ました。")
            print("batch_idx:", batch_idx)
            print("images finite:", torch.isfinite(images).all().item())
            print("labels finite:", torch.isfinite(labels).all().item())
            logits = torch.nan_to_num(
                logits,
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            )

        genre_loss = criterion(logits, labels)

        if student_features is not None and teacher_features is not None:
            distill_loss = feature_distillation_loss(
                student_features,
                teacher_features,
            )
        else:
            distill_loss = logits.new_tensor(0.0)

        loss = genre_loss + distill_alpha * distill_loss

        if not torch.isfinite(loss):
            print("loss が NaN/Inf になりました。")
            print("batch_idx:", batch_idx)
            print("images finite:", torch.isfinite(images).all().item())
            print("labels finite:", torch.isfinite(labels).all().item())
            print("logits finite:", torch.isfinite(logits).all().item())
            print("images min/max:", images.min().item(), images.max().item())
            print("labels min/max:", labels.min().item(), labels.max().item())
            print("logits min/max:", logits.min().item(), logits.max().item())
            raise ValueError("loss が NaN/Inf になりました。")

        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_genre_loss += genre_loss.item() * batch_size
        total_distill_loss += distill_loss.item() * batch_size

    dataset_size = len(loader.dataset)

    return {
        "loss": total_loss / dataset_size,
        "genre_loss": total_genre_loss / dataset_size,
        "distill_loss": total_distill_loss / dataset_size,
    }
