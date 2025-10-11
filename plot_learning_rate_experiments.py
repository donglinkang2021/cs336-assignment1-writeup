#!/usr/bin/env python3
"""
Academic-style plotting script for learning rate experiments
Generates individual PDF figures for each CSV file
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# Set up matplotlib for academic publication style
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'text.usetex': False,  # Set to True if you have LaTeX installed
    'figure.figsize': (7, 5),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight',
    'axes.linewidth': 1.2,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2.0,
    'lines.markersize': 4,
    'legend.fontsize': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
})

# Define colors for different learning rates (colorblind-friendly palette)
colors = {
    '0.03': '#d62728',    # Red - divergent
    '0.01': '#ff7f0e',    # Orange - high LR
    '0.006': '#2ca02c',   # Green - good LR
    '0.003': '#1f77b4',   # Blue - best LR
    '0.001': '#9467bd',   # Purple - medium LR
    '0.0003': '#8c564b',  # Brown - low LR
    '0.0001': '#e377c2',  # Pink - very low LR
    '3e-05': '#7f7f7f',   # Gray - very low LR
    '1e-05': '#bcbd22',   # Olive - extremely low LR
}

def extract_lr_from_column(col_name):
    """Extract learning rate value from column name"""
    if 'ts-lr' in col_name:
        lr_part = col_name.split('ts-lr')[1].split(' -')[0]
        return lr_part
    return None

def plot_learning_rate_schedule():
    """Plot three subplots: training loss, learning rate schedule, and validation loss"""
    # Read all data
    train_loss_df = pd.read_csv('exps/learning_rate/ts-train-loss.csv')
    train_lr_df = pd.read_csv('exps/learning_rate/ts-train-lr.csv')
    val_loss_df = pd.read_csv('exps/learning_rate/ts-val-loss.csv')
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    
    # Get columns
    train_loss_columns = [col for col in train_loss_df.columns if 'train/loss' in col and '__MIN' not in col and '__MAX' not in col]
    lr_columns = [col for col in train_lr_df.columns if 'train/lr' in col and '__MIN' not in col and '__MAX' not in col]
    val_loss_columns = [col for col in val_loss_df.columns if 'val/loss' in col and '__MIN' not in col and '__MAX' not in col]
    
    # Sort learning rates for consistent ordering
    lr_values = []
    for col in train_loss_columns:
        lr = extract_lr_from_column(col)
        if lr:
            lr_values.append((float(lr), col))
    lr_values.sort(reverse=True)
    
    # Store handles and labels for shared legend
    legend_handles = []
    legend_labels = []
    
    # --- LEFT SUBPLOT: Training Loss ---
    for lr_val, loss_col in lr_values:
        lr_str = str(lr_val) if lr_val >= 0.001 else f"{lr_val:.0e}"
        color = colors.get(str(lr_val), '#000000')
        
        # Get training loss data
        loss_data = train_loss_df[loss_col].dropna()
        loss_steps = train_loss_df['Step'][:len(loss_data)]
        
        # Skip if data is insufficient or diverged
        if len(loss_data) < 10 or loss_data.max() > 15:
            if lr_val >= 0.01:  # Show divergent cases briefly
                line = ax1.plot(loss_steps[:50], loss_data[:50], color=color, 
                              linestyle='--', alpha=0.7, linewidth=1.5)
                if not legend_handles:  # Only add to legend once
                    legend_handles.append(line[0])
                    legend_labels.append(f'LR={lr_str} (div.)')
            continue
        
        # Plot training loss
        line = ax1.plot(loss_steps, loss_data, color=color, linewidth=2, alpha=0.8)
        
        # Add to legend (only once)
        if len([l for l in legend_labels if lr_str in l]) == 0:
            legend_handles.append(line[0])
            legend_labels.append(f'LR={lr_str}')
    
    ax1.set_xlabel('Training Step')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 10)
    
    # --- MIDDLE SUBPLOT: Learning Rate Schedule ---
    for lr_val, _ in lr_values:
        lr_str = str(lr_val) if lr_val >= 0.001 else f"{lr_val:.0e}"
        color = colors.get(str(lr_val), '#000000')
        
        # Find corresponding lr column
        lr_col = None
        for col in lr_columns:
            if extract_lr_from_column(col) == str(lr_val):
                lr_col = col
                break
        
        if lr_col is None:
            continue
            
        # Get learning rate data
        lr_data = train_lr_df[lr_col].dropna()
        lr_steps = train_lr_df['Step'][:len(lr_data)]
        
        if len(lr_data) < 10:
            continue
            
        ax2.plot(lr_steps, lr_data, color=color, linewidth=2, alpha=0.8)
    
    ax2.set_xlabel('Training Step')
    ax2.set_title('Learning Rate Schedule')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # --- RIGHT SUBPLOT: Validation Loss (using relative time) ---
    for lr_val, _ in lr_values:
        lr_str = str(lr_val) if lr_val >= 0.001 else f"{lr_val:.0e}"
        color = colors.get(str(lr_val), '#000000')
        
        # Find corresponding validation loss column
        val_col = None
        for col in val_loss_columns:
            if extract_lr_from_column(col) == str(lr_val):
                val_col = col
                break
        
        if val_col is None:
            continue
        
        # Get valid data points using relative time
        mask = val_loss_df[val_col].notna()
        if mask.sum() < 2:
            continue
            
        # Use relative time as x-axis, convert to minutes
        times = val_loss_df['Relative Time (Process)'][mask] / 60.0  # Convert seconds to minutes
        val_loss = val_loss_df[val_col][mask]
        
        # Skip if validation loss is too high (diverged)
        if val_loss.min() > 8:
            continue
        
        ax3.plot(times, val_loss, 'o-', color=color, 
                linewidth=2, markersize=4, alpha=0.8)
    
    ax3.set_xlabel('Relative Time (min)')
    ax3.set_title('Validation Loss')
    ax3.grid(True, alpha=0.3)
    
    # Add shared legend at the bottom (south)
    fig.legend(legend_handles, legend_labels, loc='lower center', 
              bbox_to_anchor=(0.5, -0.05), ncol=len(legend_labels), 
              fontsize=10, frameon=False)
    
    # Adjust layout to make room for legend at bottom
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # Save the figure
    plt.savefig('images/learning_rate_schedule.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print("Generated: images/learning_rate_schedule.pdf")

def main():
    """Main function to generate all plots"""
    # Create images directory if it doesn't exist
    os.makedirs('images', exist_ok=True)
    
    print("Generating learning rate experiment plots...")
    
    # Generate all plots including the new combined one
    plot_learning_rate_schedule()
    
    print("Files created:")
    print("- images/learning_rate_schedule.pdf")

if __name__ == "__main__":
    main()