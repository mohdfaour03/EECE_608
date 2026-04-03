from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
from typing import Any

from dp_audit_tightness.config import TrainExperimentConfig
from dp_audit_tightness.data.datasets import DatasetBundle, load_dataset_bundle
from dp_audit_tightness.models.io import export_inference_state_dict
from dp_audit_tightness.models.simple_mlp import build_model
from dp_audit_tightness.privacy.accounting import compute_theoretical_upper_bound
from dp_audit_tightness.privacy.pld_accounting import compute_epsilon_pld
from dp_audit_tightness.utils.paths import build_run_id, ensure_directory
from dp_audit_tightness.utils.results import TrainingRunRecord


@dataclass(slots=True)
class TrainingOutcome:
    record: TrainingRunRecord
    checkpoint_path: Path | None
    model_state_dict: dict[str, Any] | None = None


def train_dp_sgd(
    config: TrainExperimentConfig,
    logger: logging.Logger,
    *,
    dataset_bundle: DatasetBundle | None = None,
    save_checkpoint: bool | None = None,
    run_descriptor: str | None = None,
    return_model_state: bool = False,
) -> TrainingOutcome:
    try:
        import torch
        from opacus import PrivacyEngine
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise RuntimeError(
            "torch, torchvision, and opacus are required for DP-SGD training. Install project dependencies first."
        ) from exc

    active_bundle = dataset_bundle or load_dataset_bundle(config.dataset, split_seed=config.run.split_seed)
    train_loader = DataLoader(
        active_bundle.train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.dataset.num_workers,
    )
    eval_loader = DataLoader(
        active_bundle.eval_dataset,
        batch_size=config.training.eval_batch_size,
        shuffle=False,
        num_workers=config.dataset.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config.model).to(device)
    optimizer = _build_optimizer(model, config)
    criterion = torch.nn.CrossEntropyLoss()

    privacy_engine = PrivacyEngine(accountant=config.privacy.accountant)
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=config.training.noise_multiplier,
        max_grad_norm=config.training.clipping_norm,
    )

    logger.info("Starting DP-SGD training.")
    for epoch_index in range(config.training.epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
        logger.info(
            "Completed epoch %s/%s with average_loss=%.4f",
            epoch_index + 1,
            config.training.epochs,
            running_loss / max(1, len(train_loader)),
        )

    utility_metrics = _evaluate_classifier(model=model, data_loader=eval_loader, device=device)
    sampling_rate = config.training.sampling_rate or (
        config.training.batch_size / active_bundle.train_size
    )
    epsilon_upper_theory = compute_theoretical_upper_bound(
        privacy_engine=privacy_engine,
        delta=config.privacy.delta,
    )

    # PLD-based accounting for a tighter upper bound.
    num_steps = config.training.epochs * len(train_loader)
    try:
        pld_result = compute_epsilon_pld(
            noise_multiplier=config.training.noise_multiplier,
            sampling_rate=sampling_rate,
            num_steps=num_steps,
            delta=config.privacy.delta,
        )
        epsilon_upper_pld = pld_result["epsilon_pld"]
        pld_backend = pld_result["backend_used"]
        logger.info(
            "PLD accounting: epsilon_pld=%.4f (backend=%s) vs RDP epsilon=%.4f",
            epsilon_upper_pld, pld_backend, epsilon_upper_theory,
        )
    except Exception as exc:
        logger.warning("PLD accounting unavailable: %s", exc)
        epsilon_upper_pld = None
        pld_backend = None

    training_run_id = build_run_id(
        prefix="train",
        descriptor=run_descriptor or config.experiment_name,
        seed=config.run.training_seed,
    )
    save_checkpoint = config.run.save_checkpoint if save_checkpoint is None else save_checkpoint
    inference_state_dict = export_inference_state_dict(model)
    checkpoint_path = None
    if save_checkpoint:
        checkpoint_dir = ensure_directory(Path(config.run.results_root) / "training" / "checkpoints")
        checkpoint_path = checkpoint_dir / f"{training_run_id}.pt"
        torch.save(inference_state_dict, checkpoint_path)

    record = TrainingRunRecord(
        training_run_id=training_run_id,
        experiment_name=config.experiment_name,
        dataset=config.dataset.name,
        split_seed=config.run.split_seed,
        training_seed=config.run.training_seed,
        model_name=config.model.name,
        optimizer_name=config.training.optimizer.name,
        learning_rate=config.training.optimizer.learning_rate,
        weight_decay=config.training.optimizer.weight_decay,
        clipping_norm=config.training.clipping_norm,
        noise_multiplier=config.training.noise_multiplier,
        batch_size=config.training.batch_size,
        epochs=config.training.epochs,
        sampling_rate=sampling_rate,
        delta=config.privacy.delta,
        epsilon_upper_theory=epsilon_upper_theory,
        utility_metrics=utility_metrics,
        model_artifact_path=str(checkpoint_path) if checkpoint_path else None,
        epsilon_upper_pld=epsilon_upper_pld,
        pld_accounting_backend=pld_backend,
        notes=config.run.notes,
    )
    return TrainingOutcome(
        record=record,
        checkpoint_path=checkpoint_path,
        model_state_dict=inference_state_dict if return_model_state else None,
    )


def _build_optimizer(model, config: TrainExperimentConfig):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for optimizer construction.") from exc

    name = config.training.optimizer.name.lower()
    if name != "sgd":
        raise NotImplementedError("The starter scaffold currently implements SGD only.")
    return torch.optim.SGD(
        model.parameters(),
        lr=config.training.optimizer.learning_rate,
        weight_decay=config.training.optimizer.weight_decay,
        momentum=config.training.optimizer.momentum,
    )


def _evaluate_classifier(model, data_loader, device) -> dict[str, float]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for model evaluation.") from exc

    model.eval()
    correct = 0
    total = 0
    cumulative_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)
            cumulative_loss += float(loss.item()) * targets.size(0)
            predictions = logits.argmax(dim=1)
            correct += int((predictions == targets).sum().item())
            total += int(targets.size(0))

    accuracy = correct / max(1, total)
    average_loss = cumulative_loss / max(1, total)
    return {"accuracy": accuracy, "loss": average_loss}
