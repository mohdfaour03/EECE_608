from __future__ import annotations


def fit_temperature(logits, labels, max_iter: int = 50) -> float:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for temperature scaling.") from exc

    if logits.numel() == 0:
        return 1.0

    temperature = torch.nn.Parameter(torch.ones((), device=logits.device))
    optimizer = torch.optim.LBFGS([temperature], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")
    criterion = torch.nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = criterion(logits / temperature.clamp(min=0.05), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(temperature.detach().clamp(min=0.05, max=10.0).item())


def apply_temperature(logits, temperature: float):
    scale = max(float(temperature), 1e-6)
    return logits / scale


def zscore_with_reference(scores, reference_scores):
    mean = reference_scores.mean()
    std = reference_scores.std(unbiased=False).clamp(min=1e-6)
    return (scores - mean) / std


def percentile_rank_transform(scores, reference_scores):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for score calibration.") from exc

    reference = reference_scores.detach().cpu()
    transformed = []
    for score in scores.detach().cpu():
        transformed.append(float((reference <= score).float().mean().item()))
    return torch.tensor(transformed, dtype=scores.dtype, device=scores.device)
