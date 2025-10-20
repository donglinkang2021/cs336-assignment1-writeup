#!/usr/bin/env python3
"""
Academic-style plotting script for batch size experiments
Generates individual PDF figures with three subplots: train loss, val loss, and GPU memory
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

# Define colors for different batch sizes (colorblind-friendly palette)
colors = {
    '1': '#e377c2',      # Pink - very small
    '2': '#bcbd22',      # Olive - very small
    '4': '#7f7f7f',      # Gray - very small
    '8': '#9467bd',      # Purple - small
    '16': '#8c564b',     # Brown - small
    '32': '#2ca02c',     # Green - medium
    '64': '#1f77b4',     # Blue - medium-large
    '128': '#ff7f0e',    # Orange - large
    '256': '#d62728',    # Red - larger
    '512': '#17becf',    # Cyan - very large
    '768': '#e7ba52',    # Yellow - very large
}

def extract_bs_from_column(col_name):
    """Extract batch size value from column name"""
    if 'ts-bs' in col_name:
        bs_part = col_name.split('ts-bs')[1].split(' -')[0]
        return bs_part
    return None

def plot_batch_size_exps(train_loss_file, val_loss_file, gpu_mem_file, output_file):
    """Plot three subplots: training loss, validation loss, and GPU memory"""
    # Read all data
    train_loss_df = pd.read_csv(train_loss_file)
    val_loss_df = pd.read_csv(val_loss_file)
    gpu_mem_df = pd.read_csv(gpu_mem_file)
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    
    # Get columns
    train_loss_columns = [col for col in train_loss_df.columns if 'train/loss' in col and '__MIN' not in col and '__MAX' not in col]
    val_loss_columns = [col for col in val_loss_df.columns if 'val/loss' in col and '__MIN' not in col and '__MAX' not in col]
    gpu_mem_columns = [col for col in gpu_mem_df.columns if 'system/gpu.0.memoryAllocatedBytes' in col and '__MIN' not in col and '__MAX' not in col]
    
    # Sort batch sizes for consistent ordering
    bs_values = []
    for col in train_loss_columns:
        bs = extract_bs_from_column(col)
        if bs:
            bs_values.append((int(bs), col))
    bs_values.sort()
    
    # Store handles and labels for shared legend
    legend_handles = []
    legend_labels = []
    
    # --- LEFT SUBPLOT: Training Loss (log-log scale) ---
    for bs_val, loss_col in bs_values:
        bs_str = str(bs_val)
        color = colors.get(bs_str, '#000000')
        
        # Get training loss data
        loss_data = train_loss_df[loss_col].dropna()
        loss_steps = train_loss_df['Step'][:len(loss_data)]
        
        # Skip if data is insufficient
        if len(loss_data) < 10:
            continue
        
        # Filter out invalid data (steps and loss must be positive for log scale)
        valid_mask = (loss_steps > 0) & (loss_data > 0)
        loss_steps_valid = loss_steps[valid_mask]
        loss_data_valid = loss_data[valid_mask]
        
        if len(loss_data_valid) < 10:
            continue
        
        # Plot training loss
        line = ax1.plot(loss_steps_valid, loss_data_valid, color=color, linewidth=2, alpha=0.8)
        
        # Add to legend
        legend_handles.append(line[0])
        legend_labels.append(f'BS={bs_str}')
    
    ax1.set_xlabel('Training Step')
    ax1.set_title('Training Loss')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, which='both')
    
    # --- MIDDLE SUBPLOT: Validation Loss (log-log scale) ---
    for bs_val, _ in bs_values:
        bs_str = str(bs_val)
        color = colors.get(bs_str, '#000000')
        
        # Find corresponding validation loss column
        val_col = None
        for col in val_loss_columns:
            if extract_bs_from_column(col) == bs_str:
                val_col = col
                break
        
        if val_col is None:
            continue
        
        # Get valid data points
        mask = val_loss_df[val_col].notna()
        if mask.sum() < 2:
            continue
        
        steps = val_loss_df['Step'][mask]
        val_loss = val_loss_df[val_col][mask]
        
        # Filter out invalid data (steps and loss must be positive for log scale)
        valid_mask = (steps > 0) & (val_loss > 0)
        steps_valid = steps[valid_mask]
        val_loss_valid = val_loss[valid_mask]
        
        if len(val_loss_valid) < 2:
            continue
        
        # Plot validation loss
        ax2.plot(steps_valid, val_loss_valid, '-', color=color, linewidth=2, alpha=0.8)
    
    ax2.set_xlabel('Training Step')
    ax2.set_title('Validation Loss')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, which='both')
    
    # --- RIGHT SUBPLOT: GPU Memory vs Relative Time ---
    for bs_val, _ in bs_values:
        bs_str = str(bs_val)
        color = colors.get(bs_str, '#000000')
        
        # Find corresponding GPU memory column
        mem_col = None
        for col in gpu_mem_columns:
            if extract_bs_from_column(col) == bs_str:
                mem_col = col
                break
        
        if mem_col is None:
            continue
        
        # Get valid data points
        mask = gpu_mem_df[mem_col].notna()
        if mask.sum() < 2:
            continue
        
        # Use relative time as x-axis, convert to minutes
        times = gpu_mem_df['Relative Time (Process)'][mask] / 60.0  # Convert seconds to minutes
        # Convert memory from bytes to GB
        memory_gb = gpu_mem_df[mem_col][mask] / (1024**3)
        
        # Plot GPU memory
        ax3.plot(times, memory_gb, '-', color=color, linewidth=2, alpha=0.8)
    
    ax3.set_xlabel('Relative Time (min)')
    ax3.set_title('GPU Memory Allocated (GB)')
    ax3.grid(True, alpha=0.3)
    
    # Add shared legend at the bottom
    fig.legend(legend_handles, legend_labels, loc='lower center', 
              bbox_to_anchor=(0.5, -0.05), ncol=len(legend_labels), 
              fontsize=10, frameon=False)
    
    # Adjust layout to make room for legend at bottom
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.subplots_adjust(bottom=0.15)
    
    # Save the figure
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Generated: {output_file}")

def main():
    """Main function to generate all plots"""
    # Create images directory if it doesn't exist
    os.makedirs('images', exist_ok=True)
    
    print("Generating batch size experiment plots...")
    
    # Generate batch size plots
    plot_batch_size_exps(
        train_loss_file='exps/batch_size/train-loss.csv',
        val_loss_file='exps/batch_size/val-loss.csv',
        gpu_mem_file='exps/batch_size/system-gpu-mem.csv',
        output_file='images/batch_size_experiments.pdf'
    )
    
    print("File created:")
    print("- images/batch_size_experiments.pdf")

if __name__ == "__main__":
    main()
