from __future__ import annotations

from pathlib import Path


def plot_tightness_curve(
    auditor_labels: list[str],
    epsilon_upper_series: list[float],
    epsilon_lower_series: list[float],
    output_path: str | Path,
) -> Path:
    """Plot theoretical upper bounds against empirical lower bounds across auditors."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting. Install project dependencies first.") from exc

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(auditor_labels, epsilon_upper_series, marker="o", label="epsilon_upper_theory")
    axis.plot(auditor_labels, epsilon_lower_series, marker="s", label="epsilon_lower_empirical")
    axis.set_ylabel("privacy loss epsilon")
    axis.set_title("Theoretical upper bounds vs empirical lower bounds")
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(target, dpi=200)
    plt.close(figure)
    return target

