from __future__ import annotations

import numpy as np
from scipy import signal, stats

from chatter_twin.models import SignalFeatures


def extract_signal_features(
    sensor_signal: np.ndarray,
    sample_rate_hz: float,
    spindle_rpm: float,
    flute_count: int,
    modal_frequency_hz: float | None = None,
) -> SignalFeatures:
    """Extract compact chatter-oriented features from a sensor window."""

    series = _as_1d_signal(sensor_signal)
    if series.size < 4:
        raise ValueError("Need at least four samples to extract features")

    rms = float(np.sqrt(np.mean(series**2)))
    peak = float(np.max(np.abs(series)))
    crest = peak / max(rms, 1.0e-12)
    kurt = float(stats.kurtosis(series, fisher=False, bias=False))

    nperseg = min(series.size, max(64, min(1024, series.size // 2)))
    frequencies, power = signal.welch(series, fs=sample_rate_hz, nperseg=nperseg)
    total_energy = float(np.trapezoid(power, frequencies)) + 1.0e-18
    tooth_frequency = spindle_rpm * flute_count / 60.0
    tooth_mask = np.zeros_like(frequencies, dtype=bool)
    band_half_width = max(5.0, 0.08 * tooth_frequency)
    for harmonic in range(1, 7):
        center = harmonic * tooth_frequency
        if center >= frequencies[-1]:
            break
        tooth_mask |= np.abs(frequencies - center) <= band_half_width
    tooth_energy = _band_energy(frequencies, power, tooth_mask)

    if modal_frequency_hz is None:
        lower = min(max(1.2 * tooth_frequency, 20.0), 0.45 * sample_rate_hz)
        upper = 0.45 * sample_rate_hz
    else:
        lower = max(5.0, 0.75 * modal_frequency_hz)
        upper = min(0.48 * sample_rate_hz, 1.25 * modal_frequency_hz)
    chatter_mask = (frequencies >= lower) & (frequencies <= upper) & ~tooth_mask
    chatter_energy = _band_energy(frequencies, power, chatter_mask)

    non_tooth_energy = max(total_energy - tooth_energy, 0.0)
    non_tooth_ratio = non_tooth_energy / max(tooth_energy, 1.0e-18)
    dominant_frequency = float(frequencies[int(np.argmax(power))])

    probability = power / (np.sum(power) + 1.0e-18)
    entropy = float(-np.sum(probability * np.log2(probability + 1.0e-18)) / np.log2(probability.size))

    return SignalFeatures(
        rms=rms,
        peak=peak,
        crest_factor=crest,
        kurtosis=kurt,
        tooth_frequency_hz=float(tooth_frequency),
        tooth_band_energy=tooth_energy,
        chatter_band_energy=chatter_energy,
        non_tooth_harmonic_ratio=float(non_tooth_ratio),
        dominant_frequency_hz=dominant_frequency,
        spectral_entropy=entropy,
        sample_rate_hz=float(sample_rate_hz),
    )


def _as_1d_signal(sensor_signal: np.ndarray) -> np.ndarray:
    array = np.asarray(sensor_signal, dtype=float)
    if array.ndim == 1:
        return array - np.mean(array)
    if array.ndim == 2:
        return np.linalg.norm(array, axis=1) - np.mean(np.linalg.norm(array, axis=1))
    raise ValueError("Sensor signal must be 1D or 2D")


def _band_energy(frequencies: np.ndarray, power: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    return float(np.trapezoid(power[mask], frequencies[mask]))
