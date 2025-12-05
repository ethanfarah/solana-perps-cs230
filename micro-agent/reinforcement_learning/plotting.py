import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Optional
import json


plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'train': '#2ecc71',
    'val': '#3498db', 
    'test': '#e74c3c',
    'equity': '#9b59b6',
    'pnl': '#1abc9c',
    'loss': '#e67e22',
}


def plot_training_curves(
    results: Dict,
    save_dir: Path
):
    epochs = [r['epoch'] for r in results['train']]
    train_equity = [r['avg_equity'] for r in results['train']]
    train_std = [r['std_equity'] for r in results['train']]
    val_equity = [r['avg_equity'] for r in results['val']]
    val_std = [r['std_equity'] for r in results['val']]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax = axes[0, 0]
    ax.plot(epochs, train_equity, 'o-', color=COLORS['train'], label='Train', linewidth=2)
    ax.fill_between(epochs, 
                    np.array(train_equity) - np.array(train_std),
                    np.array(train_equity) + np.array(train_std),
                    color=COLORS['train'], alpha=0.2)
    ax.plot(epochs, val_equity, 's-', color=COLORS['val'], label='Val', linewidth=2)
    ax.fill_between(epochs,
                    np.array(val_equity) - np.array(val_std),
                    np.array(val_equity) + np.array(val_std),
                    color=COLORS['val'], alpha=0.2)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Break-even')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Average Equity')
    ax.set_title('Train vs Validation Equity')
    ax.legend()
    
    ax = axes[0, 1]
    epsilon = [r['epsilon'] for r in results['train']]
    ax.plot(epochs, epsilon, 'o-', color=COLORS['loss'], linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Epsilon')
    ax.set_title('Exploration Rate Decay')
    
    ax = axes[1, 0]
    if 'equities' in results['train'][0]:
        train_equities_per_epoch = [r['equities'] for r in results['train']]
        positions = range(len(train_equities_per_epoch))
        bp = ax.boxplot(train_equities_per_epoch, positions=positions, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor(COLORS['train'])
            patch.set_alpha(0.6)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Equity')
        ax.set_title('Train Equity Distribution per Epoch')
    else:
        ax.text(0.5, 0.5, 'Per-chunk equities not available', 
                ha='center', va='center', transform=ax.transAxes)
    
    ax = axes[1, 1]
    if 'equities' in results['val'][0]:
        val_equities_per_epoch = [r['equities'] for r in results['val']]
        positions = range(len(val_equities_per_epoch))
        bp = ax.boxplot(val_equities_per_epoch, positions=positions, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor(COLORS['val'])
            patch.set_alpha(0.6)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Equity')
        ax.set_title('Val Equity Distribution per Epoch')
    else:
        ax.text(0.5, 0.5, 'Per-chunk equities not available',
                ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_loss_curves(
    results: Dict,
    save_dir: Path
):
    if 'avg_spot_loss' not in results['train'][0]:
        return
    
    epochs = [r['epoch'] for r in results['train']]
    spot_loss = [r.get('avg_spot_loss', 0) for r in results['train']]
    vol_loss = [r.get('avg_vol_loss', 0) for r in results['train']]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(epochs, spot_loss, 'o-', label='Spot Agent Loss', linewidth=2)
    ax.plot(epochs, vol_loss, 's-', label='Vol Agent Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Agent Losses Over Training')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / 'loss_curves.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_test_results(
    results: Dict,
    save_dir: Path
):
    if 'test' not in results:
        return
    
    test = results['test']
    equities = test.get('equities', [])
    
    if not equities:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    ax = axes[0]
    chunks = range(len(equities))
    colors = [COLORS['train'] if e >= 1.0 else COLORS['test'] for e in equities]
    ax.bar(chunks, equities, color=colors, alpha=0.7)
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=2)
    ax.axhline(y=np.mean(equities), color=COLORS['val'], linestyle='-', linewidth=2, label=f'Mean: {np.mean(equities):.4f}')
    ax.set_xlabel('Chunk')
    ax.set_ylabel('Equity')
    ax.set_title('Test Set: Equity per Chunk')
    ax.legend()
    
    ax = axes[1]
    ax.hist(equities, bins=20, color=COLORS['equity'], alpha=0.7, edgecolor='black')
    ax.axvline(x=1.0, color='gray', linestyle='--', linewidth=2, label='Break-even')
    ax.axvline(x=np.mean(equities), color=COLORS['val'], linestyle='-', linewidth=2, label=f'Mean: {np.mean(equities):.4f}')
    ax.set_xlabel('Equity')
    ax.set_ylabel('Count')
    ax.set_title('Test Set: Equity Distribution')
    ax.legend()
    
    ax = axes[2]
    cumulative = np.cumprod(equities)
    ax.plot(chunks, cumulative, '-', color=COLORS['pnl'], linewidth=2)
    ax.fill_between(chunks, 1, cumulative, where=(cumulative >= 1), color=COLORS['train'], alpha=0.3)
    ax.fill_between(chunks, 1, cumulative, where=(cumulative < 1), color=COLORS['test'], alpha=0.3)
    ax.axhline(y=1.0, color='gray', linestyle='--')
    ax.set_xlabel('Chunk')
    ax.set_ylabel('Cumulative Equity')
    ax.set_title('Test Set: Cumulative Performance')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'test_results.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_strategy_distribution(
    strategy_counts: Dict,
    save_dir: Path,
    title_prefix: str = ""
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax = axes[0]
    spot = strategy_counts['spot']
    labels = list(spot.keys())
    sizes = list(spot.values())
    
    if sum(sizes) > 0:
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, autopct='%1.1f%%',
            colors=colors, startangle=90
        )
        ax.set_title(f'{title_prefix}Spot Strategy Distribution')
    
    ax = axes[1]
    vol = strategy_counts['vol']
    labels = list(vol.keys())
    sizes = list(vol.values())
    
    if sum(sizes) > 0:
        colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, autopct='%1.1f%%',
            colors=colors, startangle=90
        )
        ax.set_title(f'{title_prefix}Vol Strategy Distribution')
    
    plt.tight_layout()
    plt.savefig(save_dir / f'{title_prefix.lower().replace(" ", "_")}strategy_distribution.png', 
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_strategy_over_time(
    strategy_history: Dict,
    save_dir: Path,
    title_prefix: str = ""
):
    if not strategy_history.get('spot_history'):
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    ax = axes[0]
    spot_history = strategy_history['spot_history']
    spot_names = list(set(spot_history))
    
    n_steps = len(spot_history)
    window = max(1, n_steps // 100)
    
    for i, name in enumerate(spot_names):
        usage = [1 if s == name else 0 for s in spot_history]
        smoothed = np.convolve(usage, np.ones(window)/window, mode='valid')
        ax.fill_between(range(len(smoothed)), 0, smoothed, alpha=0.5, label=name)
    
    ax.set_xlabel('Step')
    ax.set_ylabel('Strategy Usage')
    ax.set_title(f'{title_prefix}Spot Strategy Selection Over Time')
    ax.legend(loc='upper right', fontsize=8)
    
    ax = axes[1]
    vol_history = strategy_history['vol_history']
    vol_names = list(set(vol_history))
    
    for i, name in enumerate(vol_names):
        usage = [1 if s == name else 0 for s in vol_history]
        smoothed = np.convolve(usage, np.ones(window)/window, mode='valid')
        ax.fill_between(range(len(smoothed)), 0, smoothed, alpha=0.5, label=name)
    
    ax.set_xlabel('Step')
    ax.set_ylabel('Strategy Usage')
    ax.set_title(f'{title_prefix}Vol Strategy Selection Over Time')
    ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_dir / f'{title_prefix.lower().replace(" ", "_")}strategy_over_time.png',
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_pnl_analysis(
    pnl_history: List[float],
    position_history: List[float],
    save_dir: Path,
    title_prefix: str = ""
):
    if not pnl_history:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax = axes[0, 0]
    cumulative_pnl = np.cumsum(pnl_history)
    ax.plot(cumulative_pnl, color=COLORS['pnl'], linewidth=1)
    ax.fill_between(range(len(cumulative_pnl)), 0, cumulative_pnl,
                    where=(cumulative_pnl >= 0), color=COLORS['train'], alpha=0.3)
    ax.fill_between(range(len(cumulative_pnl)), 0, cumulative_pnl,
                    where=(cumulative_pnl < 0), color=COLORS['test'], alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--')
    ax.set_xlabel('Step')
    ax.set_ylabel('Cumulative PnL')
    ax.set_title(f'{title_prefix}Cumulative PnL')
    
    ax = axes[0, 1]
    ax.hist(pnl_history, bins=50, color=COLORS['pnl'], alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=2)
    ax.axvline(x=np.mean(pnl_history), color=COLORS['val'], linestyle='-', 
               linewidth=2, label=f'Mean: {np.mean(pnl_history):.6f}')
    ax.set_xlabel('PnL')
    ax.set_ylabel('Count')
    ax.set_title(f'{title_prefix}PnL Distribution')
    ax.legend()
    
    ax = axes[1, 0]
    ax.plot(position_history, color=COLORS['equity'], linewidth=0.5, alpha=0.7)
    ax.axhline(y=0, color='gray', linestyle='--')
    ax.set_xlabel('Step')
    ax.set_ylabel('Position')
    ax.set_title(f'{title_prefix}Position Over Time')
    
    ax = axes[1, 1]
    ax.hist(position_history, bins=50, color=COLORS['equity'], alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=2)
    ax.set_xlabel('Position')
    ax.set_ylabel('Count')
    ax.set_title(f'{title_prefix}Position Distribution')
    
    plt.tight_layout()
    plt.savefig(save_dir / f'{title_prefix.lower().replace(" ", "_")}pnl_analysis.png',
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_summary_table(
    results: Dict,
    save_dir: Path
):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    data = []
    headers = ['Metric', 'Train', 'Val', 'Test']
    
    if results['train']:
        final_train = results['train'][-1]
        final_val = results['val'][-1]
        test = results.get('test', {})
        
        data.append(['Avg Equity', 
                     f"{final_train['avg_equity']:.4f}",
                     f"{final_val['avg_equity']:.4f}",
                     f"{test.get('avg_equity', 'N/A'):.4f}" if isinstance(test.get('avg_equity'), float) else 'N/A'])
        
        data.append(['Std Equity',
                     f"{final_train['std_equity']:.4f}",
                     f"{final_val['std_equity']:.4f}",
                     f"{test.get('std_equity', 'N/A'):.4f}" if isinstance(test.get('std_equity'), float) else 'N/A'])
        
        if 'equities' in test:
            equities = test['equities']
            data.append(['Min Equity', '-', '-', f"{np.min(equities):.4f}"])
            data.append(['Max Equity', '-', '-', f"{np.max(equities):.4f}"])

    table = ax.table(cellText=data, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)
    
    for j, header in enumerate(headers):
        table[(0, j)].set_facecolor('#3498db')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    plt.title('Training Summary', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(save_dir / 'summary_table.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_strategy_combination_analysis(
    pnl_history: List[float],
    spot_history: List[str],
    vol_history: List[str],
    save_dir: Path,
    title_prefix: str = ""
):
    if not pnl_history or not spot_history or not vol_history:
        return
    
    import pandas as pd
    
    df = pd.DataFrame({
        'pnl': pnl_history,
        'spot': spot_history,
        'vol': vol_history,
    })
    
    spot_strategies = sorted(df['spot'].unique())
    vol_strategies = sorted(df['vol'].unique())
    
    mean_matrix = np.full((len(vol_strategies), len(spot_strategies)), np.nan)
    tstat_matrix = np.full((len(vol_strategies), len(spot_strategies)), np.nan)
    count_matrix = np.full((len(vol_strategies), len(spot_strategies)), 0)
    
    for i, vol_strat in enumerate(vol_strategies):
        for j, spot_strat in enumerate(spot_strategies):
            mask = (df['spot'] == spot_strat) & (df['vol'] == vol_strat)
            pnls = df.loc[mask, 'pnl'].values
            
            count_matrix[i, j] = len(pnls)
            
            if len(pnls) >= 2:
                mean_pnl = np.mean(pnls)
                std_pnl = np.std(pnls, ddof=1)
                n = len(pnls)
                
                mean_matrix[i, j] = mean_pnl
                
                if std_pnl > 0:
                    t_stat = mean_pnl / (std_pnl / np.sqrt(n))
                    tstat_matrix[i, j] = t_stat
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    max_t = np.nanmax(np.abs(tstat_matrix))
    if max_t == 0 or np.isnan(max_t):
        max_t = 3
    max_t = max(max_t, 3)
    
    im = ax.imshow(tstat_matrix, cmap='RdYlGn', aspect='auto',
                   vmin=-max_t, vmax=max_t)
    
    ax.set_xticks(range(len(spot_strategies)))
    ax.set_yticks(range(len(vol_strategies)))
    ax.set_xticklabels(spot_strategies, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(vol_strategies, fontsize=10)
    ax.set_xlabel('Spot Strategy', fontsize=12)
    ax.set_ylabel('Vol Strategy', fontsize=12)
    ax.set_title(f'{title_prefix}Strategy Combination Performance\n(colored by t-statistic)', fontsize=14)
    
    for i in range(len(vol_strategies)):
        for j in range(len(spot_strategies)):
            mean_val = mean_matrix[i, j]
            t_val = tstat_matrix[i, j]
            n_val = count_matrix[i, j]
            
            if n_val == 0:
                text = "n=0"
                color = 'gray'
            elif np.isnan(mean_val):
                text = f"n={n_val}"
                color = 'gray'
            else:
                if abs(mean_val) < 0.001:
                    mean_str = f"{mean_val:.2e}"
                else:
                    mean_str = f"{mean_val:.4f}"
                
                if np.isnan(t_val):
                    t_str = "t=N/A"
                elif abs(t_val) > 2.58:
                    t_str = f"t={t_val:.2f}**"
                elif abs(t_val) > 1.96:
                    t_str = f"t={t_val:.2f}*"
                else:
                    t_str = f"t={t_val:.2f}"
                
                text = f"μ={mean_str}\n{t_str}\nn={n_val:,}"
                
                color = 'white' if abs(t_val) > max_t * 0.4 else 'black'
            
            ax.text(j, i, text, ha='center', va='center', fontsize=8, color=color)
    
    cbar = plt.colorbar(im, ax=ax, label='T-Statistic', shrink=0.8)
    cbar.ax.axhline(y=1.96, color='black', linestyle='--', linewidth=1)
    cbar.ax.axhline(y=-1.96, color='black', linestyle='--', linewidth=1)
    cbar.ax.axhline(y=2.58, color='black', linestyle='-', linewidth=1)
    cbar.ax.axhline(y=-2.58, color='black', linestyle='-', linewidth=1)
    
    plt.tight_layout()
    plt.savefig(save_dir / f'{title_prefix.lower().replace(" ", "_")}strategy_combination_heatmap.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    ax = axes[0]
    max_abs = np.nanmax(np.abs(mean_matrix))
    if max_abs == 0 or np.isnan(max_abs):
        max_abs = 1e-6
    
    im = ax.imshow(mean_matrix, cmap='RdYlGn', aspect='auto',
                   vmin=-max_abs, vmax=max_abs)
    ax.set_xticks(range(len(spot_strategies)))
    ax.set_yticks(range(len(vol_strategies)))
    ax.set_xticklabels(spot_strategies, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(vol_strategies, fontsize=8)
    ax.set_xlabel('Spot Strategy')
    ax.set_ylabel('Vol Strategy')
    ax.set_title('Mean PnL')
    
    for i in range(len(vol_strategies)):
        for j in range(len(spot_strategies)):
            val = mean_matrix[i, j]
            if not np.isnan(val):
                color = 'white' if abs(val) > max_abs * 0.4 else 'black'
                ax.text(j, i, f'{val:.2e}', ha='center', va='center', fontsize=7, color=color)
    plt.colorbar(im, ax=ax)
    
    ax = axes[1]
    im = ax.imshow(tstat_matrix, cmap='RdYlGn', aspect='auto',
                   vmin=-max_t, vmax=max_t)
    ax.set_xticks(range(len(spot_strategies)))
    ax.set_yticks(range(len(vol_strategies)))
    ax.set_xticklabels(spot_strategies, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(vol_strategies, fontsize=8)
    ax.set_xlabel('Spot Strategy')
    ax.set_ylabel('Vol Strategy')
    ax.set_title('T-Statistic (* p<0.05, ** p<0.01)')
    
    for i in range(len(vol_strategies)):
        for j in range(len(spot_strategies)):
            val = tstat_matrix[i, j]
            if not np.isnan(val):
                color = 'white' if abs(val) > max_t * 0.4 else 'black'
                sig = '**' if abs(val) > 2.58 else ('*' if abs(val) > 1.96 else '')
                ax.text(j, i, f'{val:.2f}{sig}', ha='center', va='center', fontsize=7, color=color)
    plt.colorbar(im, ax=ax)
    
    ax = axes[2]
    im = ax.imshow(count_matrix, cmap='Blues', aspect='auto')
    ax.set_xticks(range(len(spot_strategies)))
    ax.set_yticks(range(len(vol_strategies)))
    ax.set_xticklabels(spot_strategies, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(vol_strategies, fontsize=8)
    ax.set_xlabel('Spot Strategy')
    ax.set_ylabel('Vol Strategy')
    ax.set_title('Sample Count')
    
    max_count = np.max(count_matrix)
    for i in range(len(vol_strategies)):
        for j in range(len(spot_strategies)):
            val = count_matrix[i, j]
            color = 'white' if val > max_count * 0.5 else 'black'
            ax.text(j, i, f'{val:,}', ha='center', va='center', fontsize=7, color=color)
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig(save_dir / f'{title_prefix.lower().replace(" ", "_")}strategy_combination_panels.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    
    mean_df = pd.DataFrame(mean_matrix, index=vol_strategies, columns=spot_strategies)
    tstat_df = pd.DataFrame(tstat_matrix, index=vol_strategies, columns=spot_strategies)
    count_df = pd.DataFrame(count_matrix, index=vol_strategies, columns=spot_strategies)
    
    mean_df.to_csv(save_dir / f'{title_prefix.lower().replace(" ", "_")}mean_pnl_matrix.csv')
    tstat_df.to_csv(save_dir / f'{title_prefix.lower().replace(" ", "_")}tstat_matrix.csv')
    count_df.to_csv(save_dir / f'{title_prefix.lower().replace(" ", "_")}count_matrix.csv')
    
    combined_data = []
    for i, vol_strat in enumerate(vol_strategies):
        for j, spot_strat in enumerate(spot_strategies):
            combined_data.append({
                'vol_strategy': vol_strat,
                'spot_strategy': spot_strat,
                'mean_pnl': mean_matrix[i, j],
                't_stat': tstat_matrix[i, j],
                'n': count_matrix[i, j],
                'significant_05': abs(tstat_matrix[i, j]) > 1.96 if not np.isnan(tstat_matrix[i, j]) else False,
                'significant_01': abs(tstat_matrix[i, j]) > 2.58 if not np.isnan(tstat_matrix[i, j]) else False,
            })
    
    combined_df = pd.DataFrame(combined_data)
    combined_df.to_csv(save_dir / f'{title_prefix.lower().replace(" ", "_")}strategy_combination_stats.csv', index=False)

def generate_all_plots(
    results: Dict,
    save_dir: Path,
    strategy_counts: Optional[Dict] = None,
    strategy_history: Optional[Dict] = None,
    pnl_history: Optional[List[float]] = None,
    position_history: Optional[List[float]] = None
):
    save_dir.mkdir(parents=True, exist_ok=True)
    
    plot_training_curves(results, save_dir)
    plot_loss_curves(results, save_dir)
    plot_test_results(results, save_dir)
    plot_summary_table(results, save_dir)
    
    if strategy_counts:
        plot_strategy_distribution(strategy_counts, save_dir, "Test ")
    
    if strategy_history:
        plot_strategy_over_time(strategy_history, save_dir, "Test ")
    
    if pnl_history and position_history:
        plot_pnl_analysis(pnl_history, position_history, save_dir, "Test ")
    
    if pnl_history and strategy_history:
        plot_strategy_combination_analysis(
            pnl_history,
            strategy_history['spot_history'],
            strategy_history['vol_history'],
            save_dir,
            "Test "
        )