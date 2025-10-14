#!/usr/bin/env python3
"""
Academic-style plotting script for RMSNorm ablation experiments
Generates a 2x3 grid plot with different learning rates
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# Set up matplotlib for academic publication style
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'text.usetex': False,  # Set to True if you have LaTeX installed
    'figure.figsize': (15, 8),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight',
    'axes.linewidth': 1.2,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2.0,
    'lines.markersize': 4,
    'legend.fontsize': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 12,
})

def plot_rmsnorm_ablation():
    """Plot 2x3 grid comparing baseline vs no-RMSNorm across different learning rates"""
    
    # Define learning rates to plot
    learning_rates = ['1e-2', '1e-3', '3e-4']
    lr_display = ['0.01', '0.001', '0.0003']
    
    # Create figure with 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    
    # Colors for baseline vs no-RMSNorm
    colors = {
        'baseline': '#1f77b4',      # Blue
        'no_rmsnorm': '#ff7f0e'     # Orange
    }
    
    for i, (lr, lr_disp) in enumerate(zip(learning_rates, lr_display)):
        # Read training loss data
        train_file = f'exps/ablation/remove_rmsnorm/train_loss_lr{lr}.csv'
        val_file = f'exps/ablation/remove_rmsnorm/val_loss_lr{lr}.csv'
        
        try:
            train_df = pd.read_csv(train_file)
            val_df = pd.read_csv(val_file)
        except FileNotFoundError:
            print(f"Warning: Could not find files for LR {lr}")
            continue
        
        # Top row: Training loss vs steps
        ax_train = axes[0, i]
        
        # Extract training loss columns
        baseline_train_col = None
        no_rmsnorm_train_col = None
        
        for col in train_df.columns:
            if 'baseline-ts - train/loss' in col and '__MIN' not in col and '__MAX' not in col:
                baseline_train_col = col
            elif 'baseline-ts-wo_rmsnorm - train/loss' in col and '__MIN' not in col and '__MAX' not in col:
                no_rmsnorm_train_col = col
            elif 'baseline-ts-lr0.001 - train/loss' in col and '__MIN' not in col and '__MAX' not in col:
                baseline_train_col = col
            elif 'baseline-ts-lr0.001-wo_rmsnorm - train/loss' in col and '__MIN' not in col and '__MAX' not in col:
                no_rmsnorm_train_col = col
            elif 'baseline-ts-lr0.0003 - train/loss' in col and '__MIN' not in col and '__MAX' not in col:
                baseline_train_col = col
            elif 'baseline-ts-lr0.0003-wo_rmsnorm - train/loss' in col and '__MIN' not in col and '__MAX' not in col:
                no_rmsnorm_train_col = col
        
        if baseline_train_col and no_rmsnorm_train_col:
            # Plot baseline
            baseline_data = train_df[baseline_train_col].dropna()
            steps = train_df['Step'][:len(baseline_data)]
            ax_train.plot(steps, baseline_data, color=colors['baseline'], 
                         linewidth=2, label='Baseline (with RMSNorm)', alpha=0.9)
            
            # Plot no-RMSNorm - check if it diverged
            no_rmsnorm_data = train_df[no_rmsnorm_train_col].dropna()
            steps_no_rmsnorm = train_df['Step'][:len(no_rmsnorm_data)]
            
            # If loss is extremely high, it likely diverged
            if no_rmsnorm_data.max() > 20:
                # Find where it starts to diverge
                diverge_idx = np.where(no_rmsnorm_data > 10)[0]
                if len(diverge_idx) > 0:
                    # Plot up to divergence point
                    ax_train.plot(steps_no_rmsnorm, no_rmsnorm_data, 
                                color=colors['no_rmsnorm'], linewidth=2, 
                                label='No RMSNorm (diverged)', alpha=0.9, linestyle='--')
                else:
                    ax_train.plot(steps_no_rmsnorm, no_rmsnorm_data, 
                                color=colors['no_rmsnorm'], linewidth=2, 
                                label='No RMSNorm', alpha=0.9)
            else:
                ax_train.plot(steps_no_rmsnorm, no_rmsnorm_data, 
                            color=colors['no_rmsnorm'], linewidth=2, 
                            label='No RMSNorm', alpha=0.9)
        
        ax_train.set_xlabel('Training Steps')
        ax_train.set_title(f'Training Loss (LR = {lr_disp})')
        ax_train.grid(True, alpha=0.3)
        if lr != '3e-4':
            ax_train.set_yscale('log')  # Add log scale for y-axis
        ax_train.legend(loc='upper right')
        
        
        # Bottom row: Validation loss vs relative time
        ax_val = axes[1, i]
        
        # Extract validation loss columns
        baseline_val_col = None
        no_rmsnorm_val_col = None
        
        for col in val_df.columns:
            if 'baseline-ts - val/loss' in col and '__MIN' not in col and '__MAX' not in col:
                baseline_val_col = col
            elif 'baseline-ts-wo_rmsnorm - val/loss' in col and '__MIN' not in col and '__MAX' not in col:
                no_rmsnorm_val_col = col
            elif 'baseline-ts-lr0.001 - val/loss' in col and '__MIN' not in col and '__MAX' not in col:
                baseline_val_col = col
            elif 'baseline-ts-lr0.001-wo_rmsnorm - val/loss' in col and '__MIN' not in col and '__MAX' not in col:
                no_rmsnorm_val_col = col
            elif 'baseline-ts-lr0.0003 - val/loss' in col and '__MIN' not in col and '__MAX' not in col:
                baseline_val_col = col
            elif 'baseline-ts-lr0.0003-wo_rmsnorm - val/loss' in col and '__MIN' not in col and '__MAX' not in col:
                no_rmsnorm_val_col = col
        
        if baseline_val_col and no_rmsnorm_val_col:
            # Plot baseline validation loss
            baseline_mask = val_df[baseline_val_col].notna()
            if baseline_mask.sum() > 0:
                baseline_times = val_df['Relative Time (Process)'][baseline_mask] / 60.0  # Convert to minutes
                baseline_val_loss = val_df[baseline_val_col][baseline_mask]
                ax_val.plot(baseline_times, baseline_val_loss, 'o-', color=colors['baseline'], 
                           linewidth=2, markersize=4, label='Baseline (with RMSNorm)', alpha=0.9)
            
            # Plot no-RMSNorm validation loss
            no_rmsnorm_mask = val_df[no_rmsnorm_val_col].notna()
            if no_rmsnorm_mask.sum() > 0:
                no_rmsnorm_times = val_df['Relative Time (Process)'][no_rmsnorm_mask] / 60.0
                no_rmsnorm_val_loss = val_df[no_rmsnorm_val_col][no_rmsnorm_mask]
                
                # Check if validation loss is reasonable or diverged
                if no_rmsnorm_val_loss.max() > 20:  # Extremely high values indicate divergence
                    ax_val.plot(no_rmsnorm_times, no_rmsnorm_val_loss, '--', 
                               color=colors['no_rmsnorm'], linewidth=2, markersize=4, 
                               label='No RMSNorm (diverged)', alpha=0.9)
                else:
                    ax_val.plot(no_rmsnorm_times, no_rmsnorm_val_loss, 'o-', 
                               color=colors['no_rmsnorm'], linewidth=2, markersize=4, 
                               label='No RMSNorm', alpha=0.9)
        
        ax_val.set_xlabel('Relative Time (min)')
        ax_val.set_title(f'Validation Loss (LR = {lr_disp})')
        ax_val.grid(True, alpha=0.3)
        if lr != '3e-4':
            ax_val.set_yscale('log')  # Add log scale for y-axis
        ax_val.legend(loc='upper right')
        
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    output_file = 'images/rmsnorm_ablation_experiments.pdf'
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Generated: {output_file}")

def main():
    """Main function to generate the plot"""
    # Create images directory if it doesn't exist
    os.makedirs('images', exist_ok=True)
    
    print("Generating RMSNorm ablation experiment plots...")
    plot_rmsnorm_ablation()
    print("Files created:")
    print("- images/rmsnorm_ablation_experiments.pdf")

if __name__ == "__main__":
    main()