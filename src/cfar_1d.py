"""Educational 1-D CA-CFAR demo using synthetic power samples."""
from __future__ import annotations
import numpy as np


def ca_cfar(power: np.ndarray, guard: int = 2, reference: int = 16,
            pfa: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """Return (threshold, detections); edge cells are NaN/False.

    Assumes independent exponentially distributed noise power. There are
    2*reference training cells and 2*guard+1 excluded cells.
    """
    power = np.asarray(power, dtype=float)
    if power.ndim != 1 or power.size < 2 * (guard + reference) + 1:
        raise ValueError("power must be 1-D and long enough for the window")
    if guard < 0 or reference < 1 or not 0 < pfa < 1:
        raise ValueError("invalid guard/reference/pfa")
    ntrain = 2 * reference
    alpha = ntrain * (pfa ** (-1.0 / ntrain) - 1.0)
    threshold = np.full(power.shape, np.nan)
    detections = np.zeros(power.shape, dtype=bool)
    for i in range(guard + reference, len(power) - guard - reference):
        training = np.concatenate((power[i-guard-reference:i-guard],
                                  power[i+guard+1:i+guard+reference+1]))
        threshold[i] = alpha * np.mean(training)
        detections[i] = power[i] > threshold[i]
    return threshold, detections


def demo(seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    power = rng.exponential(scale=1.0, size=256)
    target_index = 128
    target_snr_db = 12.0
    power[target_index] += 10 ** (target_snr_db / 10)
    threshold, detections = ca_cfar(power, guard=2, reference=16, pfa=1e-3)
    print(f"Target index {target_index}: detected={bool(detections[target_index])}")
    print(f"Number of detections in valid cells: {detections.sum()}")
    try:
        import matplotlib.pyplot as plt
        x = np.arange(power.size)
        plt.plot(x, power, label="Power")
        plt.plot(x, threshold, label="CA-CFAR threshold")
        plt.xlabel("Cell index"); plt.ylabel("Power")
        plt.title("Synthetic CA-CFAR demonstration"); plt.legend(); plt.tight_layout()
        plt.show()
    except ImportError:
        pass


if __name__ == "__main__":
    demo()
