from __future__ import annotations

from dp_audit_tightness.config import ModelConfig


def build_model(config: ModelConfig):
    """Central model factory.  Dispatches on ``config.name``."""
    name = config.name.lower()

    if name == "simple_mlp":
        return _build_simple_mlp(config)

    if name == "cnn_cifar10":
        from dp_audit_tightness.models.cnn import build_cnn_cifar10
        return build_cnn_cifar10(num_classes=config.num_classes)

    if name == "tabular_mlp":
        from dp_audit_tightness.models.tabular_mlp import build_tabular_mlp
        return build_tabular_mlp(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            num_classes=config.num_classes,
        )

    raise NotImplementedError(f"No model factory registered for model={config.name}")


def _build_simple_mlp(config: ModelConfig):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for model construction. Install project dependencies first.") from exc

    class SimpleMLP(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.network = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(config.input_dim, config.hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(config.hidden_dim, config.num_classes),
            )

        def forward(self, inputs):
            return self.network(inputs)

    return SimpleMLP()

