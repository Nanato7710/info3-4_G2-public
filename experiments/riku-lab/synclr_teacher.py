from pathlib import Path

import timm
import torch
from torch import nn


class SynCLRTeacher(nn.Module):
    """
    Frozen SynCLR teacher.

    This loader assumes a PyTorch checkpoint that can be mapped onto a timm
    ViT backbone. Use SynCLR ViT-B/16 first because its feature dimension is
    768, matching ConvNeXt-Tiny's classifier input.
    """

    def __init__(
        self,
        checkpoint_path: str,
        model_name: str = "vit_base_patch16_224",
        device: torch.device | None = None,
    ):
        super().__init__()

        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(
                "SynCLR checkpoint was not found: "
                f"{path}. Download a real SynCLR checkpoint, then set "
                "SYNCLR_CHECKPOINT_PATH to that file. "
                "If you want to run without SynCLR for now, set "
                "USE_SYNCLR_DISTILLATION=0."
            )

        self.model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
        )
        self.feature_dim = int(self.model.num_features)

        checkpoint = torch.load(
            path,
            map_location="cpu",
        )
        state_dict = self._extract_state_dict(checkpoint)
        filtered_state_dict = self._filter_state_dict(state_dict)

        if not filtered_state_dict:
            raise ValueError(
                "No checkpoint tensors matched the SynCLR teacher model. "
                "The checkpoint may be a JAX/Flax checkpoint or use an "
                "unsupported architecture."
            )

        load_result = self.model.load_state_dict(
            filtered_state_dict,
            strict=False,
        )

        ignored_count = len(state_dict) - len(filtered_state_dict)
        print("\n===== SynCLR teacher check =====")
        print(f"Teacher model      : {model_name}")
        print(f"Checkpoint         : {path}")
        print(f"Feature dim        : {self.feature_dim}")
        print(f"Loaded tensors     : {len(filtered_state_dict)}")
        print(f"Ignored tensors    : {ignored_count}")
        print(f"Missing tensors    : {len(load_result.missing_keys)}")
        print(f"Unexpected tensors : {len(load_result.unexpected_keys)}")

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()

        if device is not None:
            self.to(device)

    def _extract_state_dict(self, checkpoint):
        if isinstance(checkpoint, dict):
            for key in (
                "state_dict",
                "model",
                "teacher",
                "teacher_state_dict",
                "target_encoder",
                "encoder",
            ):
                value = checkpoint.get(key)
                if isinstance(value, dict):
                    return value

            if all(
                torch.is_tensor(value)
                for value in checkpoint.values()
            ):
                return checkpoint

        raise ValueError(
            "Unsupported SynCLR checkpoint format. Expected a PyTorch "
            "state_dict or a checkpoint dict containing state_dict/model."
        )

    def _filter_state_dict(self, state_dict: dict) -> dict:
        model_state = self.model.state_dict()
        filtered = {}

        for key, value in state_dict.items():
            clean_key = self._clean_key(key)

            if (
                clean_key in model_state
                and model_state[clean_key].shape == value.shape
            ):
                filtered[clean_key] = value

        return filtered

    @staticmethod
    def _clean_key(key: str) -> str:
        prefixes = (
            "module.",
            "model.",
            "backbone.",
            "encoder.",
            "visual.",
            "student.backbone.",
            "teacher.backbone.",
        )

        clean_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix):]
                    changed = True

        return clean_key

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.model.forward_features(images)

        if hasattr(self.model, "forward_head"):
            features = self.model.forward_head(
                features,
                pre_logits=True,
            )
        elif features.ndim == 3:
            features = features[:, 0]
        elif features.ndim > 2:
            features = features.flatten(1)

        return features
