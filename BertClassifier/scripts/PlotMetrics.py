import matplotlib.pyplot as plt
import numpy as np
from transformers import TrainerCallback
from pathlib import Path
from typing import Optional

class PlotMetricsCallback(TrainerCallback):
    def __init__(self, output_dir: str, smoothing_window: Optional[int] = None):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.train_losses = []
        self.eval_losses = []
        self.eval_accuracies = []
        self.train_steps = []
        self.eval_steps = []
        self.smoothing_window = smoothing_window  # None or <=1 means no smoothing

    def _smooth_data(self, data, window_size):
        """Apply moving average smoothing to reduce noise"""
        if not window_size or window_size <= 1 or len(data) < window_size:
            return data
        smoothed = []
        for i in range(len(data)):
            start = max(0, i - window_size // 2)
            end = min(len(data), i + window_size // 2 + 1)
            smoothed.append(np.mean(data[start:end]))
        return smoothed

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}

        if "loss" in logs:
            self.train_losses.append(logs["loss"])
            self.train_steps.append(state.global_step)

        if "eval_loss" in logs:
            self.eval_losses.append(logs["eval_loss"])
            self.eval_steps.append(state.global_step)

        if "eval_accuracy" in logs:
            self.eval_accuracies.append(logs["eval_accuracy"])
            if len(self.eval_steps) < len(self.eval_accuracies):
                self.eval_steps.append(state.global_step)

        if len(self.train_steps) == 0:
            return

        plt.figure(figsize=(10, 6))
        ax = plt.gca()

        use_smoothing = bool(self.smoothing_window and self.smoothing_window > 1)

        # Training loss
        y_train = self._smooth_data(self.train_losses, self.smoothing_window) if use_smoothing else self.train_losses
        ax.plot(self.train_steps, y_train, label="Training Loss" + (" (smoothed)" if use_smoothing else ""), color="#2563eb", linewidth=2)

        if len(self.eval_losses) > 0 and len(self.eval_steps) > 0:
            y_eval = self._smooth_data(self.eval_losses, self.smoothing_window) if use_smoothing else self.eval_losses
            ax.plot(self.eval_steps[:len(y_eval)], y_eval, label="Eval Loss" + (" (smoothed)" if use_smoothing else ""), color="#ef4444", linewidth=2)

        ax.set_xlabel("Steps", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.grid(True, alpha=0.2)
        ax.set_title("Training Progress", fontsize=14, fontweight='bold')

        if len(self.eval_accuracies) > 0 and len(self.eval_steps) > 0:
            ax2 = ax.twinx()
            y_acc = self._smooth_data(self.eval_accuracies, self.smoothing_window) if use_smoothing else self.eval_accuracies
            ax2.plot(self.eval_steps[:len(y_acc)], y_acc, label="Eval Accuracy" + (" (smoothed)" if use_smoothing else ""), color="#10b981", linewidth=2)
            ax2.set_ylabel("Accuracy", fontsize=12)

            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, loc="upper right", fontsize=10)
        else:
            ax.legend(loc="upper right", fontsize=10)

        plt.tight_layout()
        out_path = self.output_dir / "training_progress.png"
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
