"""Dataset preparation, subject-wise training, evaluation, and inference for GAT."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from config import GAT_CHANNEL_NAMES, GAT_CONFIG, RAW_DATA_DIR, SAMPLING_RATE
from gat_model import (
    DEPENDENCY_ERROR,
    EEGGATBiLSTMAttention,
    TORCH_GEOMETRIC_AVAILABLE,
    require_gat_dependencies,
)
from gat_utils import FEATURE_NAMES, build_channel_graph, extract_epoch_features, summarize_attention


def _extract_signal(mat_data: Dict, context: str) -> np.ndarray:
    """Read the SAM-40 Data matrix and normalize channel-first orientation."""
    candidate = mat_data.get("Data")
    if candidate is None:
        candidates = [
            value for key, value in mat_data.items()
            if not key.startswith("__") and isinstance(value, np.ndarray)
            and value.ndim == 2 and 32 in value.shape
        ]
        if len(candidates) == 1:
            candidate = candidates[0]
        else:
            raise ValueError(f"{context}: no unambiguous 32-channel EEG matrix was found (expected MAT key 'Data')")

    signal = np.asarray(candidate, dtype=np.float64)
    if signal.ndim != 2:
        raise ValueError(f"{context}: EEG data must be a 2D matrix, got shape {signal.shape}")
    if signal.shape[0] != len(GAT_CHANNEL_NAMES) and signal.shape[1] == len(GAT_CHANNEL_NAMES):
        signal = signal.T
    if signal.shape[0] != len(GAT_CHANNEL_NAMES):
        raise ValueError(
            f"{context}: expected {len(GAT_CHANNEL_NAMES)} channels, received shape {signal.shape}"
        )
    if signal.shape[1] < int(SAMPLING_RATE * GAT_CONFIG["epoch_length_seconds"]):
        raise ValueError(f"{context}: recording is shorter than one four-second EEG epoch")
    if not np.isfinite(signal).all():
        raise ValueError(f"{context}: EEG data contains NaN or infinite values")
    return signal


def preprocess_recording(raw_signal: np.ndarray, sampling_rate: int = SAMPLING_RATE) -> Tuple[np.ndarray, np.ndarray]:
    """Run repository cleaning and return ``(features, epochs)`` for a recording."""
    if sampling_rate != SAMPLING_RATE:
        raise ValueError(
            f"SAM-40 GAT preprocessing expects {SAMPLING_RATE} Hz input, received {sampling_rate} Hz"
        )
    signal = _extract_signal({"Data": raw_signal}, "EEG recording")
    try:
        from eeg_processing import EEGDataCleaner
    except ImportError as exc:
        raise ImportError("The existing EEG preprocessing module could not be imported") from exc

    cleaner = EEGDataCleaner(sampling_rate=sampling_rate)
    cleaned, _report = cleaner.clean_signal(signal)
    epochs = cleaner.segment_epochs(
        cleaned,
        epoch_length=float(GAT_CONFIG["epoch_length_seconds"]),
        overlap=float(GAT_CONFIG["epoch_overlap"]),
    )
    if epochs.ndim != 3 or epochs.shape[1] != len(GAT_CHANNEL_NAMES):
        raise ValueError(f"Preprocessing returned an unsupported EEG shape: {epochs.shape}")
    features = extract_epoch_features(epochs, sampling_rate=sampling_rate)
    return features, epochs


def build_gat_dataset(data_dir: str = RAW_DATA_DIR) -> Dict:
    """Load real SAM-40 MAT files and prepare features with subject metadata.

    Each file is a single labeled sample containing its sequence of overlapping
    four-second epochs. Keeping a trial as one sample avoids splitting highly
    overlapping epochs across train and test partitions.
    """
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise ImportError("SciPy is required to load SAM-40 .mat recordings") from exc

    from data_loader import SAM40DataLoader

    loader = SAM40DataLoader(data_dir=data_dir)
    entries = loader.scan_dataset()
    if not entries:
        raise FileNotFoundError(f"No SAM-40 .mat recordings found in: {data_dir}")

    features, labels, subjects, filenames = [], [], [], []
    skipped = []
    for entry in entries:
        filename = entry["filename"]
        if entry.get("label") not in ("normal", "stress") or not entry.get("subject_id"):
            skipped.append(f"{filename}: missing supported label or subject ID")
            continue
        try:
            mat_data = loadmat(entry["filepath"])
            raw = _extract_signal(mat_data, filename)
            trial_features, _epochs = preprocess_recording(raw)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            skipped.append(f"{filename}: {exc}")
            continue

        features.append(trial_features)
        labels.append(int(entry["label_int"]))
        subjects.append(str(entry["subject_id"]))
        filenames.append(filename)

    if not features:
        detail = "; ".join(skipped[:3])
        raise ValueError(f"No usable SAM-40 recordings were found. {detail}")
    lengths = {sample.shape[0] for sample in features}
    if len(lengths) != 1:
        raise ValueError(
            "SAM-40 recordings produced different epoch counts; this training run requires consistent "
            f"recording durations. Observed epoch counts: {sorted(lengths)}"
        )
    labels_array = np.asarray(labels, dtype=np.int64)
    if set(np.unique(labels_array).tolist()) != {0, 1}:
        raise ValueError("Training requires both normal and stressed SAM-40 recordings")

    return {
        "features": np.stack(features).astype(np.float32),
        "labels": labels_array,
        "subjects": np.asarray(subjects),
        "filenames": filenames,
        "skipped": skipped,
        "sampling_rate": SAMPLING_RATE,
        "channel_names": list(GAT_CHANNEL_NAMES),
        "feature_names": list(FEATURE_NAMES),
    }


def subject_wise_split(
    subjects: np.ndarray,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Dict[str, np.ndarray]:
    """Split entire subjects into train, validation, and test groups."""
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must be positive and leave a test split")
    subject_values = sorted(set(str(value) for value in subjects))
    if len(subject_values) < 3:
        raise ValueError("At least three distinct subjects are needed for subject-wise splitting")

    random.Random(seed).shuffle(subject_values)
    train_count = min(len(subject_values) - 2, max(1, round(len(subject_values) * train_ratio)))
    val_count = min(len(subject_values) - train_count - 1, max(1, round(len(subject_values) * val_ratio)))
    train_subjects = set(subject_values[:train_count])
    val_subjects = set(subject_values[train_count:train_count + val_count])
    test_subjects = set(subject_values[train_count + val_count:])
    subject_strings = np.asarray([str(value) for value in subjects])
    return {
        "train": np.flatnonzero(np.isin(subject_strings, list(train_subjects))),
        "validation": np.flatnonzero(np.isin(subject_strings, list(val_subjects))),
        "test": np.flatnonzero(np.isin(subject_strings, list(test_subjects))),
        "train_subjects": sorted(train_subjects),
        "validation_subjects": sorted(val_subjects),
        "test_subjects": sorted(test_subjects),
    }


def _classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> Dict:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    predictions = (probabilities >= 0.5).astype(np.int64)
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }
    metrics["roc_auc"] = (
        float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else None
    )
    return metrics


def _evaluate(model, features: np.ndarray, labels: np.ndarray, mean: np.ndarray, std: np.ndarray,
              batch_size: int, device) -> Dict:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    scaled = (features - mean.reshape(1, 1, 1, -1)) / std.reshape(1, 1, 1, -1)
    dataset = TensorDataset(torch.from_numpy(scaled), torch.from_numpy(labels.astype(np.float32)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    all_probabilities, all_labels = [], []
    criterion = torch.nn.BCEWithLogitsLoss()
    loss_total = 0.0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(x_batch)
            loss_total += float(criterion(logits, y_batch).item()) * len(y_batch)
            all_probabilities.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            all_labels.extend(y_batch.cpu().numpy().astype(np.int64).tolist())
    result = _classification_metrics(np.asarray(all_labels), np.asarray(all_probabilities))
    result["loss"] = loss_total / max(1, len(all_labels))
    return result


def train_gat_model(
    data_dir: str = RAW_DATA_DIR,
    config: Optional[Dict] = None,
    callback: Optional[Callable[[int, int, Dict], None]] = None,
    model_path: Optional[str] = None,
    metrics_path: Optional[str] = None,
) -> Dict:
    """Train, validate, test, and save the real GAT model on SAM-40."""
    require_gat_dependencies()
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    settings = dict(GAT_CONFIG)
    if config:
        settings.update(config)
    if int(settings["layers"]) != 2:
        raise ValueError("This GAT implementation has two trainable GATConv layers")
    if int(settings["epochs"]) < 1 or int(settings["batch_size"]) < 1:
        raise ValueError("epochs and batch_size must be positive")

    seed = int(settings["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    dataset = build_gat_dataset(data_dir)
    splits = subject_wise_split(dataset["subjects"], seed=seed)
    for split_name in ("train", "validation", "test"):
        if len(splits[split_name]) == 0:
            raise ValueError(f"The subject-wise {split_name} split is empty")
        split_labels = dataset["labels"][splits[split_name]]
        if len(np.unique(split_labels)) < 2:
            raise ValueError(
                f"The subject-wise {split_name} split has only one class. "
                "Use more balanced subject coverage before training."
            )

    train_x = dataset["features"][splits["train"]]
    train_y = dataset["labels"][splits["train"]]
    val_x = dataset["features"][splits["validation"]]
    val_y = dataset["labels"][splits["validation"]]
    test_x = dataset["features"][splits["test"]]
    test_y = dataset["labels"][splits["test"]]

    # Fit normalization only on the training subjects, then reuse for val/test.
    feature_mean = train_x.mean(axis=(0, 1, 2)).astype(np.float32)
    feature_std = train_x.std(axis=(0, 1, 2)).astype(np.float32)
    feature_std[feature_std < 1e-6] = 1.0
    train_x = (train_x - feature_mean.reshape(1, 1, 1, -1)) / feature_std.reshape(1, 1, 1, -1)
    val_x = (val_x - feature_mean.reshape(1, 1, 1, -1)) / feature_std.reshape(1, 1, 1, -1)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y.astype(np.float32))),
        batch_size=int(settings["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, graph_edges = build_channel_graph()
    model_settings = {
        "hidden_dim": int(settings["hidden_dim"]),
        "heads": int(settings["heads"]),
        "dropout": float(settings["dropout"]),
    }
    model = EEGGATBiLSTMAttention(
        edge_index=graph_edges,
        input_features=len(dataset["feature_names"]),
        **model_settings,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["learning_rate"]))
    loss_function = torch.nn.BCEWithLogitsLoss()
    history = {"loss": [], "val_loss": [], "val_accuracy": []}
    best_val_loss = float("inf")
    best_state = None

    for epoch_index in range(int(settings["epochs"])):
        model.train()
        running_loss = 0.0
        sample_count = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_batch)
            loss = loss_function(logits, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(y_batch)
            sample_count += len(y_batch)

        train_loss = running_loss / max(1, sample_count)
        validation = _evaluate(
            model,
            dataset["features"][splits["validation"]],
            val_y,
            feature_mean,
            feature_std,
            int(settings["batch_size"]),
            device,
        )
        history["loss"].append(train_loss)
        history["val_loss"].append(validation["loss"])
        history["val_accuracy"].append(validation["accuracy"])
        if validation["loss"] < best_val_loss:
            best_val_loss = validation["loss"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if callback:
            callback(epoch_index + 1, int(settings["epochs"]), {
                "loss": train_loss,
                "val_loss": validation["loss"],
                "val_accuracy": validation["accuracy"],
            })

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    test_metrics = _evaluate(model, test_x, test_y, feature_mean, feature_std,
                             int(settings["batch_size"]), device)

    checkpoint_path = model_path or settings["model_path"]
    results_path = metrics_path or settings["metrics_path"]
    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(results_path)), exist_ok=True)
    checkpoint = {
        "state_dict": best_state,
        "model_config": model_settings,
        "input_features": len(dataset["feature_names"]),
        "feature_mean": torch.as_tensor(feature_mean),
        "feature_std": torch.as_tensor(feature_std),
        "channel_names": dataset["channel_names"],
        "feature_names": dataset["feature_names"],
        "sampling_rate": dataset["sampling_rate"],
        "graph_edge_index": torch.as_tensor(graph_edges, dtype=torch.long),
        "sequence_steps": int(dataset["features"].shape[1]),
        "training_epochs": int(settings["epochs"]),
        "best_validation_loss": best_val_loss,
    }
    temporary_path = checkpoint_path + ".tmp"
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, checkpoint_path)

    metrics = {
        "model": "GAT + BiLSTM + Attention",
        "training_run_note": (
            "One-epoch smoke check; not a tuned performance run."
            if int(settings["epochs"]) == 1 else "Training run; assess using the held-out subject test metrics."
        ),
        "evaluation_source": "Real SAM-40 MATLAB recordings",
        "split_method": "Subject-wise 80/10/10 split",
        "seed": seed,
        "subjects": {
            "train": splits["train_subjects"],
            "validation": splits["validation_subjects"],
            "test": splits["test_subjects"],
        },
        "sample_counts": {
            "train": int(len(splits["train"])),
            "validation": int(len(splits["validation"])),
            "test": int(len(splits["test"])),
        },
        "config": {
            **settings,
            "model_path": checkpoint_path,
            "metrics_path": results_path,
        },
        "training_history": history,
        "test": test_metrics,
        "skipped_files": dataset["skipped"],
    }
    with open(results_path, "w", encoding="utf-8") as results_file:
        json.dump(metrics, results_file, indent=2)
    return metrics


def load_trained_gat(model_path: Optional[str] = None):
    """Load a saved GAT checkpoint or raise an actionable error."""
    require_gat_dependencies()
    import torch

    checkpoint_path = model_path or GAT_CONFIG["model_path"]
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"No trained GAT model exists at {checkpoint_path}. "
            "Train it from Model Training → GAT + BiLSTM first."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as exc:
        raise RuntimeError(f"Could not load the trained GAT checkpoint: {exc}") from exc

    required = {"state_dict", "model_config", "feature_mean", "feature_std", "graph_edge_index"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"GAT checkpoint is missing required entries: {', '.join(sorted(missing))}")
    model = EEGGATBiLSTMAttention(
        edge_index=checkpoint["graph_edge_index"],
        input_features=int(checkpoint["input_features"]),
        **checkpoint["model_config"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    checkpoint["feature_mean"] = checkpoint["feature_mean"].detach().cpu().numpy().astype(np.float32)
    checkpoint["feature_std"] = checkpoint["feature_std"].detach().cpu().numpy().astype(np.float32)
    return model, checkpoint, device


def predict_gat_recording(model, checkpoint: Dict, device, raw_signal: np.ndarray) -> Dict:
    """Predict one real 32-channel recording and expose model attention."""
    import torch

    features, epochs = preprocess_recording(raw_signal, int(checkpoint["sampling_rate"]))
    mean = checkpoint["feature_mean"].reshape(1, 1, 1, -1)
    std = checkpoint["feature_std"].reshape(1, 1, 1, -1)
    normalized = (features[None, ...] - mean) / std
    x = torch.from_numpy(normalized.astype(np.float32)).to(device)
    model.eval()
    with torch.no_grad():
        logits, attention = model(x, return_attention=True)
        probability = float(torch.sigmoid(logits)[0].cpu().item())
        channel_scores = attention["channel_attention"].cpu().numpy()
        edge_index = attention["edge_index"].cpu().numpy()
        edge_weights = attention["edge_attention"].cpu().numpy()
        temporal_weights = attention["temporal_attention"][0].cpu().numpy()

    channels, relationships = summarize_attention(edge_index, edge_weights)
    overlap_step = float(GAT_CONFIG["epoch_length_seconds"]) * (1.0 - float(GAT_CONFIG["epoch_overlap"]))
    epoch_length = float(GAT_CONFIG["epoch_length_seconds"])
    important_windows = []
    for epoch_idx, weights in enumerate(temporal_weights):
        start = epoch_idx * overlap_step
        important_windows.append({
            "start_seconds": start,
            "end_seconds": start + epoch_length,
            "attention": float(weights),
        })
    important_windows.sort(key=lambda item: item["attention"], reverse=True)

    return {
        "prediction": "STRESSED" if probability >= 0.5 else "NORMAL",
        "stress_probability": probability,
        "confidence": max(probability, 1.0 - probability),
        "channels": channels,
        "channel_attention": channel_scores,
        "relationships": relationships,
        "edge_index": edge_index,
        "edge_attention": edge_weights,
        "temporal_windows": important_windows,
        "num_channels": len(GAT_CHANNEL_NAMES),
        "num_graph_nodes": len(GAT_CHANNEL_NAMES),
        "num_epochs": int(epochs.shape[0]),
        "sampling_rate": int(checkpoint["sampling_rate"]),
        "model_config": checkpoint["model_config"],
    }


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the SAM-40 GAT + BiLSTM model")
    parser.add_argument("--data-dir", default=RAW_DATA_DIR, help="Directory containing SAM-40 .mat files")
    parser.add_argument("--epochs", type=int, default=GAT_CONFIG["epochs"])
    parser.add_argument("--batch-size", type=int, default=GAT_CONFIG["batch_size"])
    parser.add_argument("--learning-rate", type=float, default=GAT_CONFIG["learning_rate"])
    parser.add_argument("--hidden-dim", type=int, default=GAT_CONFIG["hidden_dim"])
    parser.add_argument("--heads", type=int, default=GAT_CONFIG["heads"])
    parser.add_argument("--dropout", type=float, default=GAT_CONFIG["dropout"])
    parser.add_argument("--seed", type=int, default=GAT_CONFIG["seed"])
    parser.add_argument("--model-path", default=GAT_CONFIG["model_path"])
    parser.add_argument("--metrics-path", default=GAT_CONFIG["metrics_path"])
    args = parser.parse_args()
    settings = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "hidden_dim": args.hidden_dim,
        "heads": args.heads,
        "dropout": args.dropout,
        "seed": args.seed,
    }

    def show_epoch(epoch: int, total: int, metrics: Dict) -> None:
        print(
            f"Epoch {epoch:03}/{total}: loss={metrics['loss']:.4f}, "
            f"val_loss={metrics['val_loss']:.4f}, val_accuracy={metrics['val_accuracy']:.4f}"
        )

    result = train_gat_model(args.data_dir, settings, show_epoch, args.model_path, args.metrics_path)
    print(f"Test metrics: {json.dumps(result['test'], indent=2)}")
    print(f"Saved model: {args.model_path}")
    print(f"Saved test metrics: {args.metrics_path}")
    if result["skipped_files"]:
        print(f"Skipped {len(result['skipped_files'])} invalid recording(s):")
        for error in result["skipped_files"][:10]:
            print(f"  - {error}")


if __name__ == "__main__":
    if not TORCH_GEOMETRIC_AVAILABLE:
        raise SystemExit(
            "PyTorch/PyTorch Geometric are required. "
            "Install dependencies with `python -m pip install -r requirements.txt`. "
            f"Import detail: {DEPENDENCY_ERROR}"
        )
    _cli()
