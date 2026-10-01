import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path


def plot_training_metrics(output_dir="output", save_path="training_loss_plot.png"):
    """
    Generează un grafic cu training și evaluation loss din ultimul checkpoint.
    
    Args:
        output_dir: Directorul cu checkpoint-urile
        save_path: Calea unde să salveze graficul
    """
    # Găsește ultimul checkpoint
    output_path = Path(output_dir)
    checkpoints = sorted([d for d in output_path.glob("checkpoint-*") if d.is_dir()], 
                        key=lambda x: x.stat().st_mtime)
    
    if not checkpoints:
        print(f"Nu s-au găsit checkpoint-uri în {output_dir}")
        return
    
    latest_checkpoint = checkpoints[-1]
    trainer_state_path = latest_checkpoint / "trainer_state.json"
    
    if not trainer_state_path.exists():
        print(f"Nu s-a găsit trainer_state.json în {latest_checkpoint}")
        return
    
    # Citește trainer_state.json
    with open(trainer_state_path, 'r') as f:
        trainer_state = json.load(f)
    
    # Extrage datele din log_history
    train_steps = []
    train_losses = []
    eval_steps = []
    eval_losses = []
    
    # New metrics
    learning_rates = []
    grad_norms = []
    
    for entry in trainer_state['log_history']:
        if 'loss' in entry:  # Training loss
            train_steps.append(entry['step'])
            train_losses.append(entry['loss'])
            if 'learning_rate' in entry:
                learning_rates.append(entry['learning_rate'])
            if 'grad_norm' in entry:
                grad_norms.append(entry['grad_norm'])
        elif 'eval_loss' in entry:  # Evaluation loss
            # Skip NaN values
            if not np.isnan(entry['eval_loss']):
                eval_steps.append(entry['step'])
                eval_losses.append(entry['eval_loss'])
    
    # Funcție pentru moving average (smoothing)
    def smooth_curve(values, window_size=20):
        """Aplică moving average pentru a reduce zgomotul"""
        if len(values) < window_size:
            return values
        smoothed = []
        for i in range(len(values)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(values), i + window_size // 2 + 1)
            smoothed.append(sum(values[start_idx:end_idx]) / (end_idx - start_idx))
        return smoothed
    
    # Smooth training loss pentru grafic mai clar
    train_losses_smooth = smooth_curve(train_losses, window_size=min(50, len(train_losses) // 4))
    
    # Create 2x2 grid of plots
    fig, axs = plt.subplots(2, 2, figsize=(20, 14))
    
    # --- Plot 1: Loss (Top Left) ---
    ax_loss = axs[0, 0]
    # Plot training loss raw (transparent)
    ax_loss.plot(train_steps, train_losses, label='Training Loss (raw)', color='lightblue', 
            linewidth=1, alpha=0.3, linestyle='--')
    # Plot training loss smoothed (principal)
    ax_loss.plot(train_steps, train_losses_smooth, label='Training Loss (smoothed)', 
            color='blue', linewidth=2.5, alpha=0.9)
    # Plot evaluation loss
    if eval_losses:
        ax_loss.plot(eval_steps, eval_losses, label='Evaluation Loss', color='red', 
                linewidth=2.5, marker='o', markersize=6, alpha=0.8, linestyle='-')
        # Add vertical lines for eval points
        for step in eval_steps:
            ax_loss.axvline(x=step, color='gray', linestyle='--', alpha=0.3)
    
    ax_loss.set_xlabel('Steps', fontsize=11)
    ax_loss.set_ylabel('Loss', fontsize=11)
    ax_loss.set_title('Training vs Evaluation Loss', fontsize=13, fontweight='bold')
    ax_loss.legend(fontsize=10, loc='best')
    ax_loss.grid(True, alpha=0.3, linestyle='--')
    ax_loss.set_ylim(bottom=0)
    
    # --- Plot 2: Learning Rate (Top Right) ---
    ax_lr = axs[0, 1]
    if learning_rates and len(learning_rates) == len(train_steps):
        ax_lr.plot(train_steps, learning_rates, color='purple', linewidth=2)
        ax_lr.set_xlabel('Steps', fontsize=11)
        ax_lr.set_ylabel('Learning Rate', fontsize=11)
        ax_lr.set_title('Learning Rate Schedule', fontsize=13, fontweight='bold')
        ax_lr.grid(True, alpha=0.3, linestyle='--')
        # Formatted Y labels
        ax_lr.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    else:
        ax_lr.text(0.5, 0.5, 'LR data mismatch or missing', ha='center', va='center')

    # --- Plot 3: Gradient Norm (Bottom Left) ---
    ax_grad = axs[1, 0]
    if grad_norms and len(grad_norms) == len(train_steps):
        # Smooth grad norms too
        grad_smooth = smooth_curve(grad_norms, window_size=min(20, len(grad_norms)//2))
        ax_grad.plot(train_steps, grad_norms, color='green', alpha=0.3, linewidth=1, label='Raw')
        ax_grad.plot(train_steps, grad_smooth, color='darkgreen', linewidth=2, label='Smoothed')
        ax_grad.set_xlabel('Steps', fontsize=11)
        ax_grad.set_ylabel('Gradient Norm', fontsize=11)
        ax_grad.set_title('Gradient Norm Stability', fontsize=13, fontweight='bold')
        ax_grad.grid(True, alpha=0.3, linestyle='--')
        ax_grad.legend()
    else:
        ax_grad.text(0.5, 0.5, 'Grad Norm data missing', ha='center', va='center')

    # --- Plot 4: Generalization Gap (Bottom Right) ---
    ax_gap = axs[1, 1]
    if eval_losses:
        # Interpolate training loss to match eval steps for comparison
        train_interp = np.interp(eval_steps, train_steps, train_losses_smooth)
        gen_gap = np.array(eval_losses) - train_interp
        
        colors = ['red' if x > 0 else 'green' for x in gen_gap]
        ax_gap.bar(eval_steps, gen_gap, width=(eval_steps[1]-eval_steps[0])*0.8 if len(eval_steps)>1 else 100,
                  color=colors, alpha=0.7)
        ax_gap.axhline(0, color='black', linewidth=1)
        ax_gap.set_xlabel('Steps', fontsize=11)
        ax_gap.set_ylabel('Diff (Eval - Train)', fontsize=11)
        ax_gap.set_title('Generalization Gap (Overfitting Indicator)', fontsize=13, fontweight='bold')
        ax_gap.grid(True, alpha=0.3, linestyle='--')
        
        # Add text explaining colors
        ax_gap.text(0.02, 0.95, 'Red = Overfitting\nGreen = Underfitting', 
                   transform=ax_gap.transAxes, fontsize=9,
                   bbox=dict(facecolor='white', alpha=0.8))
    else:
         ax_gap.text(0.5, 0.5, 'Requires Evaluation Data', ha='center', va='center')
    
    plt.tight_layout()
    
    # Salvează graficul
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Graficul a fost salvat ca '{save_path}'")
    
    # Afișează statistici
    print(f"\n📊 Statistici Training:")
    print(f"  - Loss inițial: {train_losses[0]:.4f}")
    print(f"  - Loss final: {train_losses[-1]:.4f}")
    print(f"  - Loss minim: {min(train_losses):.4f}")
    print(f"  - Total steps: {train_steps[-1]}")
    print(f"  - Total epochs: {trainer_state['epoch']:.2f}")
    
    if eval_losses:
        print(f"\n📊 Statistici Evaluation:")
        print(f"  - Loss minim: {min(eval_losses):.4f}")
        print(f"  - Număr evaluări: {len(eval_losses)}")
    
    plt.close()
    
    return save_path


if __name__ == "__main__":
    plot_training_metrics()
