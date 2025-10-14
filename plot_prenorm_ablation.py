#!/usr/bin/env python3
"""
Academic-style plotting script for pre-norm vs post-norm ablation experiments
Generates a 1x2 grid comparing training and validation loss
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

EXPECTED_COLUMNS = {
    'lr = 0.01': {
        'train': {
            'pre': 'baseline-ts - train/loss',
            'post': 'baseline-ts-prenorm2postnorm - train/loss',
        },
        'val': {
            'pre': 'baseline-ts - val/loss',
            'post': 'baseline-ts-prenorm2postnorm - val/loss',
        },
        'color': '#1f77b4',
    },
    'lr = 0.001': {
        'train': {
            'pre': 'baseline-ts-lr0.001 - train/loss',
            'post': 'baseline-ts-lr0.001-prenorm2postnorm - train/loss',
        },
        'val': {
            'pre': 'baseline-ts-lr0.001 - val/loss',
            'post': 'baseline-ts-lr0.001-prenorm2postnorm - val/loss',
        },
        'color': '#2ca02c',
    },
    'lr = 0.0003': {
        'train': {
            'pre': 'baseline-ts-lr0.0003 - train/loss',
            'post': 'baseline-ts-lr0.0003-prenorm2postnorm - train/loss',
        },
        'val': {
            'pre': 'baseline-ts-lr0.0003 - val/loss',
            'post': 'baseline-ts-lr0.0003-prenorm2postnorm - val/loss',
        },
        'color': '#ff7f0e',
    },
}
LINESTYLES = {'pre': '-', 'post': '--'}


def _find_column(columns, include_tokens, exclude_tokens=None):
    """Return the first column containing all include tokens and none of the exclude tokens."""
    exclude_tokens = exclude_tokens or []
    for column in columns:
        if all(token in column for token in include_tokens) and not any(
            token in column for token in exclude_tokens
        ):
            return column
    return None


def plot_prenorm_ablation():
    """Plot pre-norm vs post-norm performance for training and validation."""
    train_file = Path('exps/ablation/prenorm2postnorm/train_loss.csv')
    val_file = Path('exps/ablation/prenorm2postnorm/val_loss.csv')

    if not train_file.exists() or not val_file.exists():
        raise FileNotFoundError('Expected CSV logs missing from exps/ablation/prenorm2postnorm')

    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    ax_train, ax_val = axes
    ax_train.set_xlabel('Training Steps')
    ax_train.set_title('Training Loss')

    ax_val.set_xlabel('Relative Time (min)')
    ax_val.set_title('Validation Loss')

    for lr_label, spec in EXPECTED_COLUMNS.items():
        train_cols = spec['train']
        val_cols = spec['val']
        color = spec['color']

        for regime, col_name in train_cols.items():
            column = _find_column(train_df.columns, [col_name], ['__MIN', '__MAX']) or _find_column(
                train_df.columns, [col_name.split(' - ')[0], 'train/loss'], ['__MIN', '__MAX']
            )
            if not column:
                raise ValueError(f'Missing training column for {lr_label} ({regime})')
            mask = train_df[column].notna()
            ax_train.plot(
                train_df.loc[mask, 'Step'],
                train_df.loc[mask, column],
                linestyle=LINESTYLES[regime],
                color=color,
                alpha=0.9,
                label=f'{lr_label} {"Pre" if regime == "pre" else "Post"}-Norm',
            )

        for regime, col_name in val_cols.items():
            column = _find_column(val_df.columns, [col_name], ['__MIN', '__MAX']) or _find_column(
                val_df.columns, [col_name.split(' - ')[0], 'val/loss'], ['__MIN', '__MAX']
            )
            if not column:
                raise ValueError(f'Missing validation column for {lr_label} ({regime})')
            mask = val_df[column].notna()
            ax_val.plot(
                val_df.loc[mask, 'Relative Time (Process)'] / 60.0,
                val_df.loc[mask, column],
                linestyle=LINESTYLES[regime],
                color=color,
                alpha=0.9,
                marker='o',
                markersize=3,
                label=f'{lr_label} {"Pre" if regime == "pre" else "Post"}-Norm',
            )
    handles, labels = [], []
    seen = set()
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in seen:
                handles.append(handle)
                labels.append(label)
                seen.add(label)
    fig.legend(
        handles,
        labels,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        fontsize=10,
        frameon=False,
    )

    plt.tight_layout()

    output_path = Path('images/prenorm_ablation_experiments.pdf')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f'Generated: {output_path}')


def main():
    """Entry point to generate the pre/post-norm ablation figure."""
    os.makedirs('images', exist_ok=True)
    print('Generating pre-norm vs post-norm ablation plots...')
    plot_prenorm_ablation()
    print('Files created:')
    print('- images/prenorm_ablation_experiments.pdf')


if __name__ == '__main__':
    main()
