from __future__ import annotations

import pandas as pd


class MetricTracker:
    """
    Aggregates metrics across many batches.

    Supports two scopes:
      - epoch: cumulative for the whole epoch
      - window: cumulative since last reset_window()
    """

    def __init__(self, *keys: str, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(
            index=list(keys),
            columns=[
                "total",
                "counts",
                "average",  # epoch
                "w_total",
                "w_counts",
                "w_average",  # window
            ],
        )
        self.reset()

    def reset(self):
        """Reset both epoch and window accumulators."""
        for col in self._data.columns:
            self._data.loc[:, col] = 0

    def reset_window(self):
        """Reset only window accumulators."""
        for col in ["w_total", "w_counts", "w_average"]:
            self._data.loc[:, col] = 0

    def update(self, key: str, value: float, n: int = 1):
        if key not in self._data.index:
            self._data.loc[key] = [0.0] * len(self._data.columns)

        # epoch
        self._data.loc[key, "total"] += value * n  # type: ignore
        self._data.loc[key, "counts"] += n  # type: ignore
        self._data.loc[key, "average"] = self._data.total[key] / self._data.counts[key]

        # window
        self._data.loc[key, "w_total"] += value * n  # type: ignore
        self._data.loc[key, "w_counts"] += n  # type: ignore
        self._data.loc[key, "w_average"] = (
            self._data.w_total[key] / self._data.w_counts[key]
        )

    def avg(self, key: str) -> float:
        """Epoch average."""
        return float(self._data.average[key])

    def avg_window(self, key: str) -> float:
        """Window average (since last reset_window)."""
        return float(self._data.w_average[key])

    def result(self) -> dict[str, float]:
        """Epoch averages for all metrics."""
        return {k: float(v) for k, v in self._data["average"].to_dict().items()} # type: ignore

    def result_window(self) -> dict[str, float]:
        """Window averages for all metrics."""
        return {k: float(v) for k, v in self._data["w_average"].to_dict().items()} # type: ignore

    def count(self, key: str) -> int:
        """Number of updates for this metric in the current epoch."""
        return int(self._data.counts.get(key, 0))

    def count_window(self, key: str) -> int:
        """Number of updates for this metric since last reset_window()."""
        return int(self._data.w_counts.get(key, 0))

    def keys(self):
        return self._data.index
