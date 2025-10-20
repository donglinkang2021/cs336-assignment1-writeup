#!/usr/bin/env python3
"""
Academic-style plotting script for main experiments comparing TinyStories and OpenWebText
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

DATASETS = {
    'TinyStories': {
        'train_col': 'baseline-ts (Run set 2) - train/loss',
        'val_col': 'baseline-ts (Run set 2) - val/loss',
        'color': '#1f77b4',
        'label': 'TinyStories',
    },
    'OpenWebText': {
        'train_col': 'main-owt (Run set) - train/loss',
        'val_col': 'main-owt (Run set) - val/loss',
        'color': '#ff7f0e',
        'label': 'OpenWebText',
    },
}


def _find_column(columns, include_tokens, exclude_tokens=None):
    """Return the first column containing all include tokens and none of the exclude tokens."""
    exclude_tokens = exclude_tokens or []
    for column in columns:
        if all(token in column for token in include_tokens) and not any(
            token in column for token in exclude_tokens
        ):
            return column
    return None


def plot_main_experiments():
    """Plot TinyStories vs. OpenWebText performance for training and validation."""
    train_file = Path('exps/main/train_loss.csv')
    val_file = Path('exps/main/val_loss.csv')

    if not train_file.exists() or not val_file.exists():
        raise FileNotFoundError('Expected CSV logs missing from exps/main')

    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    ax_train, ax_val = axes
    
    ax_train.set_xlabel('Training Steps')
    ax_train.set_title('Training Loss')

    ax_val.set_xlabel('Training Steps')
    ax_val.set_title('Validation Loss')

    # Plot training curves
    for dataset_name, spec in DATASETS.items():
        train_col = spec['train_col']
        val_col = spec['val_col']
        color = spec['color']
        label = spec['label']

        # Find training column
        column = _find_column(train_df.columns, [train_col], ['__MIN', '__MAX'])
        if not column:
            raise ValueError(f'Missing training column for {dataset_name}')
        
        mask = train_df[column].notna()
        ax_train.plot(
            train_df.loc[mask, 'Step'],
            train_df.loc[mask, column],
            color=color,
            alpha=0.9,
            label=label,
        )

        # Find validation column
        column = _find_column(val_df.columns, [val_col], ['__MIN', '__MAX'])
        if not column:
            raise ValueError(f'Missing validation column for {dataset_name}')
        
        mask = val_df[column].notna()
        ax_val.plot(
            val_df.loc[mask, 'Step'],
            val_df.loc[mask, column],
            color=color,
            alpha=0.9,
            marker='o',
            markersize=3,
            label=label,
        )

    # Add legends to each subplot
    ax_train.legend(loc='upper right', frameon=True, framealpha=0.9)
    ax_val.legend(loc='upper right', frameon=True, framealpha=0.9)

    plt.tight_layout()

    output_path = Path('images/main_experiments.pdf')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f'Generated: {output_path}')


def main():
    """Entry point to generate the main experiments figure."""
    os.makedirs('images', exist_ok=True)
    print('Generating main experiments plots (TinyStories vs. OpenWebText)...')
    plot_main_experiments()
    print('Files created:')
    print('- images/main_experiments.pdf')


if __name__ == '__main__':
    main()
