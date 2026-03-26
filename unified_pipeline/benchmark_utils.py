import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


METRIC_COLUMNS = [
    "branch",
    "model",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "model_size_mb",
    "trainable_params",
    "fps",
    "inference_time_ms",
    "train_time_sec",
    "train_samples",
    "train_samples_per_sec",
    "sec_per_epoch",
    "map50",
    "map50_95",
    "box_loss",
    "cls_loss",
    "dfl_loss",
    "notes",
]


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def compute_f1(precision: Optional[float], recall: Optional[float]) -> Optional[float]:
    p = safe_float(precision)
    r = safe_float(recall)
    if p is None or r is None or (p + r) == 0:
        return None
    return 2.0 * p * r / (p + r)


def bytes_to_mb(size_bytes: int) -> float:
    return float(size_bytes) / (1024.0 * 1024.0)


def count_files_recursive(root: Path, patterns: List[str]) -> int:
    total = 0
    for pattern in patterns:
        total += len(list(root.rglob(pattern)))
    return total


def benchmark_callable(
    infer_fn: Callable[[], None],
    warmup_iters: int = 3,
    timed_iters: int = 10,
    samples_per_iter: int = 1,
) -> Dict[str, Optional[float]]:
    warmup_iters = max(0, int(warmup_iters))
    timed_iters = max(1, int(timed_iters))
    samples_per_iter = max(1, int(samples_per_iter))

    for _ in range(warmup_iters):
        infer_fn()

    start = time.perf_counter()
    for _ in range(timed_iters):
        infer_fn()
    elapsed = time.perf_counter() - start

    if elapsed <= 0:
        return {"fps": None, "inference_time_ms": None}

    total_samples = timed_iters * samples_per_iter
    fps = total_samples / elapsed
    inference_time_ms = (elapsed / total_samples) * 1000.0
    return {"fps": fps, "inference_time_ms": inference_time_ms}


def parse_last_losses_from_results_csv(results_csv: Path) -> Dict[str, Optional[float]]:
    if not results_csv.exists():
        return {"box_loss": None, "cls_loss": None, "dfl_loss": None}

    with results_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return {"box_loss": None, "cls_loss": None, "dfl_loss": None}

    last = rows[-1]

    def pick(*keys: str) -> Optional[float]:
        for k in keys:
            if k in last:
                v = safe_float(last.get(k))
                if v is not None:
                    return v
        return None

    return {
        "box_loss": pick("train/box_loss", "box_loss", "loss_box"),
        "cls_loss": pick("train/cls_loss", "cls_loss", "loss_cls"),
        "dfl_loss": pick("train/dfl_loss", "dfl_loss", "loss_dfl"),
    }


def write_metrics_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def append_metrics_csv(csv_path: Path, row: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    row_out = {k: row.get(k) for k in METRIC_COLUMNS}
    file_exists = csv_path.exists()

    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_out)
