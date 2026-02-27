"""
Figure 1 — Optimizer × Capacity
Softmax accuracy at test length 301 across embedding sizes (10 seeds each).
SGD plateaus ~95% regardless of capacity; AdamW peaks at d≤52 then degrades.

Data source: run main.py with --optim AdamW/SGD across --emb-size values,
collect softmax_acc from results dict at test_seq_len=301.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Raw per-seed results at test length 301 ──────────────────────────────────


adamw_raw = {
    1:    [100.0] * 10,
    15:   [100.0] * 10,
    52:   [100.0] * 10,
    112:  [100.0, 95.5, 100.0, 95.0, 100.0, 95.0, 95.5, 100.0, 95.0, 100.0],
    512:  [100.0, 100.0, 100.0, 46.0, 100.0, 100.0, 100.0, 83.0, 54.0, 100.0],
    1024: [100.0, 93.5, 95.0, 54.0, 100.0, 54.0, 46.0, 46.0, 46.0, 100.0],
}

# SGD: means and std-devs read from original bar-chart figure
sgd_data = {
    1:    (49.9, 1.0),
    15:   (92.0, 5.5),
    32:   (95.3, 2.5),
    112:  (89.8, 3.5),
    512:  (93.5, 3.0),
    1024: (94.6, 2.5),
}

# ── Derived stats ─────────────────────────────────────────────────────────────
adamw_sizes = [1, 15, 52, 112, 512, 1024]
adamw_means = [np.mean(adamw_raw[d]) for d in adamw_sizes]
adamw_stds  = [np.std(adamw_raw[d])  for d in adamw_sizes]

sgd_sizes = [1, 15, 32, 112, 512, 1024]
sgd_means = [sgd_data[d][0] for d in sgd_sizes]
sgd_stds  = [sgd_data[d][1] for d in sgd_sizes]

# ── Shared x-axis (all embedding sizes, evenly spaced) ───────────────────────
x_labels = [1, 15, 32, 52, 112, 512, 1024]
x_pos    = {d: i for i, d in enumerate(x_labels)}

adamw_x = [x_pos[d] for d in adamw_sizes]
sgd_x   = [x_pos[d] for d in sgd_sizes]

# ── Plot ──────────────────────────────────────────────────────────────────────
blue = '#3b7dc8'
red  = '#b83030'

fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=150)
ax.set_facecolor('#f8f8f8')
fig.patch.set_facecolor('white')
ax.grid(True, color='white', linewidth=1.2, zorder=0)
for spine in ax.spines.values():
    spine.set_visible(False)

# AdamW
ax.plot(adamw_x, adamw_means, color=blue, linewidth=2,
        marker='o', markersize=5, label='AdamW', zorder=3)
ax.fill_between(adamw_x,
    np.array(adamw_means) - np.array(adamw_stds),
    np.array(adamw_means) + np.array(adamw_stds),
    color=blue, alpha=0.15, zorder=2)

# SGD
ax.plot(sgd_x, sgd_means, color=red, linewidth=2, linestyle='--',
        marker='s', markersize=5, label='SGD', zorder=3)
ax.fill_between(sgd_x,
    np.array(sgd_means) - np.array(sgd_stds),
    np.array(sgd_means) + np.array(sgd_stds),
    color=red, alpha=0.15, zorder=2)

ax.set_xticks(list(range(len(x_labels))))
ax.set_xticklabels([str(d) for d in x_labels], fontsize=9)
ax.set_xlabel('Embedding size  $d$', fontsize=10, labelpad=6)
ax.set_ylabel('Softmax accuracy at test length 301 (%)', fontsize=10, labelpad=6)
ax.set_ylim(40, 105)
ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
ax.tick_params(axis='both', labelsize=9, length=0)
ax.set_title('Softmax accuracy at maximum test length', fontsize=10, pad=10)
ax.legend(fontsize=9, framealpha=0.9, edgecolor='#cccccc', loc='lower right')

plt.tight_layout()
plt.savefig('fig1_optimizer_capacity.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
