#!/usr/bin/env python3
"""
Academic-style plotting script for leaderboard experiments
Generates a 1x2 grid showing training and validation loss vs. relative time
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Configure matplotlib for report-ready figures
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'text.usetex': False,
    'figure.figsize': (10, 4.5),
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


def _find_column(columns, include_tokens, exclude_tokens=None):
    """Return the first column containing all include tokens and none of the exclude tokens."""
    exclude_tokens = exclude_tokens or []
    for column in columns:
        if all(token in column for token in include_tokens) and not any(
            token in column for token in exclude_tokens
        ):
            return column
    return None


def plot_leaderboard_experiments():
    """Plot leaderboard training and validation loss vs. relative time."""
    train_file = Path('exps/leaderboard/train-loss.csv')
    val_file = Path('exps/leaderboard/val-loss.csv')

    if not train_file.exists() or not val_file.exists():
        raise FileNotFoundError('Expected CSV logs missing from exps/leaderboard')

    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)

    # Find the loss columns (excluding __MIN and __MAX)
    train_loss_col = _find_column(train_df.columns, ['train/loss'], ['__MIN', '__MAX'])
    val_loss_col = _find_column(val_df.columns, ['val/loss'], ['__MIN', '__MAX'])

    if not train_loss_col or not val_loss_col:
        raise ValueError('Could not find loss columns in CSV files')

    # For training data, we need to convert steps to relative time
    # Estimate from validation data which has both steps and time
    if 'Relative Time (Process)' in val_df.columns:
        step_col = _find_column(val_df.columns, ['_step'], ['__MIN', '__MAX'])
        if step_col is None:
            raise ValueError(f'Could not find step column in validation data. Available columns: {val_df.columns.tolist()}')
        val_steps = val_df[step_col].values
        val_times = val_df['Relative Time (Process)'].values
        
        # Linear interpolation for training steps
        train_steps = train_df['Step'].values
        train_times = []
        for step in train_steps:
            # Find the closest validation points
            if step <= val_steps[0]:
                # Extrapolate backwards
                time = val_times[0] * (step / val_steps[0])
            elif step >= val_steps[-1]:
                # Extrapolate forwards
                time_per_step = (val_times[-1] - val_times[0]) / (val_steps[-1] - val_steps[0])
                time = val_times[-1] + (step - val_steps[-1]) * time_per_step
            else:
                # Interpolate
                idx = 0
                for i in range(len(val_steps) - 1):
                    if val_steps[i] <= step <= val_steps[i + 1]:
                        idx = i
                        break
                t0, t1 = val_times[idx], val_times[idx + 1]
                s0, s1 = val_steps[idx], val_steps[idx + 1]
                time = t0 + (t1 - t0) * (step - s0) / (s1 - s0)
            train_times.append(time)
        
        train_df['Relative Time'] = train_times
    else:
        raise ValueError('Cannot find time information in validation data')

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    ax_train, ax_val = axes
    
    # Plot training loss vs. relative time
    mask = train_df[train_loss_col].notna()
    # Convert seconds to minutes
    train_times_min = train_df.loc[mask, 'Relative Time'] / 60
    ax_train.plot(
        train_times_min,
        train_df.loc[mask, train_loss_col],
        color='#1f77b4',
        alpha=0.9,
        label='Training Loss',
    )
    ax_train.set_xlabel('Relative Time (min)')
    ax_train.set_title('Training Loss')
    
    # Add 90 minute (1.5 hour) reference line and value for training
    if train_times_min.max() >= 90:
        # Interpolate loss value at 90 minutes
        train_times_arr = train_times_min.values
        train_loss_arr = train_df.loc[mask, train_loss_col].values
        loss_at_90 = None
        for i in range(len(train_times_arr) - 1):
            if train_times_arr[i] <= 90 <= train_times_arr[i + 1]:
                # Linear interpolation
                t0, t1 = train_times_arr[i], train_times_arr[i + 1]
                l0, l1 = train_loss_arr[i], train_loss_arr[i + 1]
                loss_at_90 = l0 + (l1 - l0) * (90 - t0) / (t1 - t0)
                break
        
        if loss_at_90 is not None:
            ax_train.axvline(x=90, color='red', linestyle='--', alpha=0.5)
            ax_train.plot(90, loss_at_90, 'ro', markersize=6, zorder=5)
            ax_train.text(92, loss_at_90 * 1.03, f'{loss_at_90:.3f}', 
                         verticalalignment='bottom', horizontalalignment='left',
                         fontsize=9, color='red')
    
    ax_train.legend(loc='upper right', frameon=True, framealpha=0.9)
    
    # Plot validation loss vs. relative time
    mask = val_df[val_loss_col].notna()
    # Convert seconds to minutes
    val_times_min = val_df.loc[mask, 'Relative Time (Process)'] / 60
    ax_val.plot(
        val_times_min,
        val_df.loc[mask, val_loss_col],
        color='#ff7f0e',
        alpha=0.9,
        marker='o',
        markersize=4,
        label='Validation Loss',
    )
    ax_val.set_xlabel('Relative Time (min)')
    ax_val.set_title('Validation Loss')
    
    # Add 90 minute (1.5 hour) reference line and value for validation
    if val_times_min.max() >= 90:
        # Find closest validation point to 90 minutes
        val_times_arr = val_times_min.values
        val_loss_arr = val_df.loc[mask, val_loss_col].values
        loss_at_90 = None
        
        # Find the closest point or interpolate
        if 90 in val_times_arr:
            idx = list(val_times_arr).index(90)
            loss_at_90 = val_loss_arr[idx]
        else:
            for i in range(len(val_times_arr) - 1):
                if val_times_arr[i] <= 90 <= val_times_arr[i + 1]:
                    # Linear interpolation
                    t0, t1 = val_times_arr[i], val_times_arr[i + 1]
                    l0, l1 = val_loss_arr[i], val_loss_arr[i + 1]
                    loss_at_90 = l0 + (l1 - l0) * (90 - t0) / (t1 - t0)
                    break
        
        if loss_at_90 is not None:
            ax_val.axvline(x=90, color='red', linestyle='--', alpha=0.5)
            ax_val.plot(90, loss_at_90, 'ro', markersize=6, zorder=5)
            ax_val.text(92, loss_at_90 * 1.01, f'{loss_at_90:.3f}', 
                       verticalalignment='bottom', horizontalalignment='left',
                       fontsize=9, color='red')
    
    ax_val.legend(loc='upper right', frameon=True, framealpha=0.9)

    plt.tight_layout()

    output_path = Path('images/leaderboard_experiments.pdf')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f'Generated: {output_path}')


def main():
    """Entry point to generate the leaderboard experiments figure."""
    os.makedirs('images', exist_ok=True)
    print('Generating leaderboard experiments plots...')
    plot_leaderboard_experiments()
    print('Files created:')
    print('- images/leaderboard_experiments.pdf')


if __name__ == '__main__':
    main()
