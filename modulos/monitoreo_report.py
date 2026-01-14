"""Monitoring helpers for notebook section 5."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Monitoreo:
    """Helper class for training monitoring charts."""

    def __init__(self, base_dir: str = "monitoreo") -> None:
        self.base_dir = base_dir

    def _load(self, relative_path: str) -> np.ndarray:
        return np.load(f"{self.base_dir}/{relative_path}", allow_pickle=True)

    def _plot_comparison(self, results: dict, test_cols: list[str], val_cols: list[str]) -> pd.DataFrame:
        df_results = pd.DataFrame(results)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

        df_results[test_cols].plot(
            ax=axes[0],
            marker="o",
            style=["r", "g"],
        )
        axes[0].set_title("Testing Accuracy")
        axes[0].set_xlabel("Run")
        axes[0].set_ylabel("Accuracy")
        axes[0].grid(True)

        df_results[val_cols].plot(
            ax=axes[1],
            marker="o",
            style=["r", "g"],
        )
        axes[1].set_title("Validation Accuracy")
        axes[1].set_xlabel("Run")
        axes[1].grid(True)

        plt.suptitle("TF vs GEN – Accuracy Comparison", fontsize=14)
        plt.tight_layout()
        plt.show()

        return df_results

    def compare_01_vs_02(self) -> pd.DataFrame:
        results = {
            "01 Testing": self._load("01_init_gen/test_accs.npy"),
            "02 Testing": self._load("02_init_tf/test_accs.npy"),
            "01 Validation": self._load("01_init_gen/val_accs.npy"),
            "02 Validation": self._load("02_init_tf/val_accs.npy"),
        }
        return self._plot_comparison(results, ["01 Testing", "02 Testing"], ["01 Validation", "02 Validation"])

    def compare_02_vs_03(self) -> pd.DataFrame:
        results = {
            "02 Testing": self._load("02_init_tf/test_accs.npy"),
            "03 Testing": self._load("03_scratch_aug/test_accs.npy"),
            "02 Validation": self._load("02_init_tf/val_accs.npy"),
            "03 Validation": self._load("03_scratch_aug/val_accs.npy"),
        }
        return self._plot_comparison(results, ["03 Testing", "02 Testing"], ["03 Validation", "02 Validation"])

    def compare_03_vs_04(self) -> pd.DataFrame:
        results = {
            "04 Testing": self._load("04_scratch_lr3e_4/test_accs.npy"),
            "03 Testing": self._load("03_scratch_aug/test_accs.npy"),
            "04 Validation": self._load("04_scratch_lr3e_4/val_accs.npy"),
            "03 Validation": self._load("03_scratch_aug/val_accs.npy"),
        }
        return self._plot_comparison(results, ["03 Testing", "04 Testing"], ["03 Validation", "04 Validation"])

    def compare_04_vs_05(self) -> pd.DataFrame:
        results = {
            "04 Testing": self._load("04_scratch_lr3e_4/test_accs.npy"),
            "05 Testing": self._load("05_scratch_l2-1e4/test_accs.npy"),
            "04 Validation": self._load("04_scratch_lr3e_4/val_accs.npy"),
            "05 Validation": self._load("05_scratch_l2-1e4/val_accs.npy"),
        }
        return self._plot_comparison(results, ["05 Testing", "04 Testing"], ["05 Validation", "04 Validation"])

    def compare_05_vs_06(self) -> pd.DataFrame:
        results = {
            "06 Testing": self._load("06_scratch_smooth/test_accs.npy"),
            "05 Testing": self._load("05_scratch_l2-1e4/test_accs.npy"),
            "06 Validation": self._load("06_scratch_smooth/val_accs.npy"),
            "05 Validation": self._load("05_scratch_l2-1e4/val_accs.npy"),
        }
        return self._plot_comparison(results, ["05 Testing", "06 Testing"], ["05 Validation", "06 Validation"])

    def _load_all_results(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        test_results = {
            "01 Init GEN": self._load("01_init_gen/test_accs.npy"),
            "02 Init TF": self._load("02_init_tf/test_accs.npy"),
            "03 Scratch + Aug": self._load("03_scratch_aug/test_accs.npy"),
            "04 Scratch + LR": self._load("04_scratch_lr3e_4/test_accs.npy"),
            "05 Scratch + L2": self._load("05_scratch_l2-1e4/test_accs.npy"),
            "06 Scratch + LS": self._load("06_scratch_smooth/test_accs.npy"),
        }

        val_results = {
            "01 Init GEN": self._load("01_init_gen/val_accs.npy"),
            "02 Init TF": self._load("02_init_tf/val_accs.npy"),
            "03 Scratch + Aug": self._load("03_scratch_aug/val_accs.npy"),
            "04 Scratch + LR": self._load("04_scratch_lr3e_4/val_accs.npy"),
            "05 Scratch + L2": self._load("05_scratch_l2-1e4/val_accs.npy"),
            "06 Scratch + LS": self._load("06_scratch_smooth/val_accs.npy"),
        }

        return pd.DataFrame(test_results), pd.DataFrame(val_results)

    def show_mean_std_summary(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        df_test, df_val = self._load_all_results()

        test_mean = df_test.mean()
        test_std = df_test.std()

        val_mean = df_val.mean()
        val_std = df_val.std()

        colors = {
            "test_mean": "#1f77b4",
            "test_std": "#ff7f0e",
            "val_mean": "#2ca02c",
            "val_std": "#d62728",
        }

        x = range(1, len(test_mean) + 1)
        xticks = list(x)

        fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)

        axes[0, 0].plot(x, test_mean.values, marker="o", color=colors["test_mean"])
        axes[0, 0].set_title("Test Accuracy – Mean")
        axes[0, 0].set_ylabel("Accuracy")
        axes[0, 0].grid(True)

        axes[0, 1].plot(x, test_std.values, marker="o", color=colors["test_std"])
        axes[0, 1].set_title("Test Accuracy – Std")
        axes[0, 1].set_ylabel("Std Dev")
        axes[0, 1].grid(True)

        axes[1, 0].plot(x, val_mean.values, marker="o", color=colors["val_mean"])
        axes[1, 0].set_title("Validation Accuracy – Mean")
        axes[1, 0].set_ylabel("Accuracy")
        axes[1, 0].set_xlabel("Model Number")
        axes[1, 0].grid(True)

        axes[1, 1].plot(x, val_std.values, marker="o", color=colors["val_std"])
        axes[1, 1].set_title("Validation Accuracy – Std")
        axes[1, 1].set_ylabel("Std Dev")
        axes[1, 1].set_xlabel("Model Number")
        axes[1, 1].grid(True)

        for ax in axes.flatten():
            ax.set_xticks(xticks)

        plt.suptitle("Comparación de Estrategias – Media y Estabilidad", fontsize=14)
        plt.tight_layout()
        plt.show()

        return df_test, df_val

    def show_min_max_summary(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        df_test, df_val = self._load_all_results()

        test_max = df_test.max()
        test_min = df_test.min()

        val_max = df_val.max()
        val_min = df_val.min()

        x = range(1, len(test_max) + 1)
        xticks = list(x)

        colors = {
            "test_max": "#9467bd",
            "test_min": "#8c564b",
            "val_max": "#17becf",
            "val_min": "#7f7f7f",
        }

        fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)

        axes[0, 0].plot(x, test_max.values, marker="o", color=colors["test_max"])
        axes[0, 0].set_title("Test Accuracy – Max")
        axes[0, 0].set_ylabel("Accuracy")
        axes[0, 0].grid(True)

        axes[0, 1].plot(x, test_min.values, marker="o", color=colors["test_min"])
        axes[0, 1].set_title("Test Accuracy – Min")
        axes[0, 1].set_ylabel("Accuracy")
        axes[0, 1].grid(True)

        axes[1, 0].plot(x, val_max.values, marker="o", color=colors["val_max"])
        axes[1, 0].set_title("Validation Accuracy – Max")
        axes[1, 0].set_ylabel("Accuracy")
        axes[1, 0].set_xlabel("Model Number")
        axes[1, 0].grid(True)

        axes[1, 1].plot(x, val_min.values, marker="o", color=colors["val_min"])
        axes[1, 1].set_title("Validation Accuracy – Min")
        axes[1, 1].set_ylabel("Accuracy")
        axes[1, 1].set_xlabel("Model Number")
        axes[1, 1].grid(True)

        for ax in axes.flatten():
            ax.set_xticks(xticks)

        plt.suptitle("Comparación de Estrategias – Mejores y Peores Casos", fontsize=14)
        plt.tight_layout()
        plt.show()

        return df_test, df_val
