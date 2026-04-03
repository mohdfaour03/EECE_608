from __future__ import annotations

from typing import Any, Mapping

from dp_audit_tightness.config import ModelConfig
from dp_audit_tightness.models.simple_mlp import build_model


def load_model_for_inference(
    model_config: ModelConfig,
    checkpoint_path: str,
    device=None,
):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for model loading.") from exc

    resolved_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_config).to(resolved_device)
    state_dict = torch.load(checkpoint_path, map_location=resolved_device)
    model.load_state_dict(normalize_state_dict_keys(state_dict))
    model.eval()
    return model


def load_model_from_state_dict(
    model_config: ModelConfig,
    state_dict: Mapping[str, Any],
    device=None,
):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for model loading.") from exc

    resolved_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_config).to(resolved_device)
    model.load_state_dict(normalize_state_dict_keys(state_dict))
    model.eval()
    return model


def export_inference_state_dict(model) -> dict[str, Any]:
    return normalize_state_dict_keys(model.state_dict())


def normalize_state_dict_keys(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in state_dict.items():
        clean_key = key.removeprefix("_module.")
        normalized[clean_key] = value
    return normalized

