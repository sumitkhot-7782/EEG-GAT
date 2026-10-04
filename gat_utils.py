"""Graph topology and EEG feature helpers for the SAM-40 GAT model.

Edges describe the neighborhood topology supplied to the model. They are not
measurements of physiological connectivity.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from config import GAT_CHANNEL_NAMES, FREQUENCY_BANDS, SAMPLING_RATE


CHANNEL_POSITIONS: Dict[str, Tuple[float, float]] = {
    "Fp1": (-0.30, 0.94), "Fp2": (0.30, 0.94),
    "F7": (-0.78, 0.65), "F3": (-0.34, 0.61), "Fz": (0.0, 0.66),
    "F4": (0.34, 0.61), "F8": (0.78, 0.65),
    "FC5": (-0.73, 0.32), "FC3": (-0.35, 0.32),
    "FC4": (0.35, 0.32), "FC6": (0.73, 0.32),
    "T3": (-0.98, 0.03), "C3": (-0.35, 0.03), "Cz": (0.0, 0.03),
    "C4": (0.35, 0.03), "T4": (0.98, 0.03),
    "TP7": (-0.91, -0.28), "CP5": (-0.72, -0.30),
    "CP3": (-0.35, -0.30), "CP4": (0.35, -0.30),
    "CP6": (0.72, -0.30), "TP8": (0.91, -0.28),
    "P7": (-0.78, -0.59), "P3": (-0.35, -0.58),
    "Pz": (0.0, -0.62), "P4": (0.35, -0.58), "P8": (0.78, -0.59),
    "PO3": (-0.32, -0.84), "PO4": (0.32, -0.84),
    "O1": (-0.34, -0.99), "Oz": (0.0, -1.0), "O2": (0.34, -0.99),
}

FEATURE_NAMES = [
    "mean", "standard_deviation", "rms", "line_length", "peak_to_peak",
    "delta_power", "theta_power", "alpha_power", "beta_power", "gamma_power",
]


def build_channel_graph(k_neighbors: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Return channel coordinates and bidirectional k-nearest-neighbor edges."""
    if k_neighbors < 1 or k_neighbors >= len(GAT_CHANNEL_NAMES):
        raise ValueError("k_neighbors must be between 1 and 31")

    coordinates = np.asarray([CHANNEL_POSITIONS[name] for name in GAT_CHANNEL_NAMES], dtype=np.float32)
    distances = np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)

    undirected = set()
    for source in range(len(GAT_CHANNEL_NAMES)):
        for target in np.argsort(distances[source])[:k_neighbors]:
            undirected.add((min(source, int(target)), max(source, int(target))))

    directed_edges = []
    for source, target in sorted(undirected):
        directed_edges.extend(((source, target), (target, source)))
    edge_index = np.asarray(directed_edges, dtype=np.int64).T
    return coordinates, edge_index


def extract_epoch_features(
    epochs: np.ndarray,
    sampling_rate: int = SAMPLING_RATE,
) -> np.ndarray:
    """Extract ten time/frequency features per channel and four-second epoch.

    Args:
        epochs: Float array shaped ``(epochs, 32 channels, samples)``.
    Returns:
        Float32 array shaped ``(epochs, 32 channels, 10 features)``.
    """
    signal = np.asarray(epochs, dtype=np.float64)
    if signal.ndim != 3:
        raise ValueError(f"Expected epochs with shape (epochs, channels, samples), got {signal.shape}")
    if signal.shape[1] != len(GAT_CHANNEL_NAMES):
        raise ValueError(f"Expected {len(GAT_CHANNEL_NAMES)} channels, got {signal.shape[1]}")
    if signal.shape[2] < 16:
        raise ValueError("Each EEG epoch must contain at least 16 samples")
    if not np.isfinite(signal).all():
        raise ValueError("EEG epochs contain NaN or infinite values")

    means = signal.mean(axis=2)
    stds = signal.std(axis=2)
    rms = np.sqrt(np.mean(np.square(signal), axis=2))
    line_length = np.mean(np.abs(np.diff(signal, axis=2)), axis=2)
    peak_to_peak = np.ptp(signal, axis=2)

    spectrum = np.abs(np.fft.rfft(signal, axis=2)) ** 2 / (signal.shape[2] ** 2)
    frequencies = np.fft.rfftfreq(signal.shape[2], d=1.0 / sampling_rate)
    band_powers = []
    for low, high in FREQUENCY_BANDS.values():
        mask = (frequencies >= low) & (frequencies < high)
        if not np.any(mask):
            band_powers.append(np.zeros(signal.shape[:2], dtype=np.float64))
        else:
            power = spectrum[:, :, mask].sum(axis=2)
            band_powers.append(np.log1p(power))

    features = np.stack(
        [means, stds, rms, line_length, peak_to_peak, *band_powers], axis=-1
    )
    return features.astype(np.float32)


def summarize_attention(
    edge_index: np.ndarray,
    edge_weights: Sequence[float],
    channel_names: Iterable[str] = GAT_CHANNEL_NAMES,
) -> Tuple[List[Dict], List[Dict]]:
    """Create normalized node and undirected-edge importance summaries."""
    names = list(channel_names)
    edge_index = np.asarray(edge_index, dtype=np.int64)
    weights = np.asarray(edge_weights, dtype=np.float64)
    if edge_index.shape[0] != 2 or edge_index.shape[1] != len(weights):
        raise ValueError("Attention values do not match the graph edge list")

    node_score = np.zeros(len(names), dtype=np.float64)
    node_count = np.zeros(len(names), dtype=np.float64)
    undirected: Dict[Tuple[int, int], List[float]] = {}
    for (source, target), weight in zip(edge_index.T, weights):
        if source == target:
            continue
        node_score[target] += weight
        node_count[target] += 1
        key = (min(int(source), int(target)), max(int(source), int(target)))
        undirected.setdefault(key, []).append(float(weight))

    node_score /= np.maximum(node_count, 1)
    total = node_score.sum()
    if total > 0:
        node_score /= total
    channels = [
        {"channel": names[index], "attention": float(node_score[index])}
        for index in np.argsort(node_score)[::-1]
    ]

    relationships = []
    for (source, target), values in undirected.items():
        relationships.append({
            "source": names[source],
            "target": names[target],
            "attention": float(np.mean(values)),
        })
    relationships.sort(key=lambda item: item["attention"], reverse=True)
    return channels, relationships
