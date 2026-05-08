"""
Trajectory Analysis for DNT Cells

This script performs trajectory inference and pseudotime analysis on T cell subsets,
with a focus on IL10+EOMES+ DNT cells. It includes:
- PAGA connectivity analysis at major cell type level
- PAGA analysis stratified by compartment (PBMC vs tissue)
- DNT-specific pseudotime trajectory from PBMC to tissue
- Gene expression dynamics along the pseudotime trajectory

Author: [Your Name]
Date: 2026-05-08
"""

import scanpy as sc
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

sc.settings.verbosity = 2
sc.settings.set_figure_params(dpi=150, frameon=False, fontsize=12)

# ============================================================
# Part 1: Configuration and Paths
# ============================================================
SCORED_PATH = '/public/home/chenjiaminggroup/yinjinwen/DNT-scRNA/IBD_PBMC/02_cross_compartment/pbmc_tissue_T_migration_scored.h5ad'
RAW_PATH = '/public/home/chenjiaminggroup/yinjinwen/DNT-scRNA/IBD_PBMC/01_processed/pbmc_tissueBGI_merge_T_cells_DNT_labeled.h5ad'
OUT_DIR = '/public/home/chenjiaminggroup/yinjinwen/DNT-scRNA/IBD_PBMC/02_cross_compartment'
os.makedirs(OUT_DIR, exist_ok=True)

if os.path.exists(SCORED_PATH):
    DATA_PATH = SCORED_PATH
    print(f'Using scored data: {SCORED_PATH}')
else:
    DATA_PATH = RAW_PATH
    print(f'Using raw data: {RAW_PATH}')

# ============================================================
# Part 2: Load and Normalize Data
# ============================================================
print('Loading data...')
adata = sc.read_h5ad(DATA_PATH)
print(f'Loaded: {adata.shape}')

# Normalize if needed
if adata.X.max() > 20:
    print('Normalizing...')
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

# ============================================================
# Part 3: Define Major Cell Types
# ============================================================
if 'major_type' not in adata.obs.columns:
    subtype_to_major = {
        'CD4 Tn': 'CD4 T', 'CD4 Tcm': 'CD4 T', 'CD4 CTL': 'CD4 T',
        'CD4 Tem': 'CD4 T', 'CD4+ Treg': 'Treg', 'Treg': 'Treg',
        'CD8 Tn': 'CD8 T', 'CD8 Tcm': 'CD8 T', 'CD8 Tem': 'CD8 T',
        'CD8 CTL': 'CD8 T', 'CD8 TEMRA': 'CD8 T', 'CD8 MAIT': 'CD8 T',
        'γδ T': 'γδT', 'γδT': 'γδT', 'gdT': 'γδT',
        'NKT': 'NKT', 'MKI67+ T': 'MKI67+ T',
        'IL10+EOMES+ DNT': 'DNT',
    }
    adata.obs['major_type'] = adata.obs['subtype_anno_2'].map(
        lambda x: subtype_to_major.get(x, 'Other T')
    )
    mask_dnt = adata.obs['is_DNT'] == True
    adata.obs.loc[mask_dnt, 'major_type'] = 'DNT'

mask_dnt = adata.obs['major_type'] == 'IL10+EOMES+ DNT'
print(f'\nDNT: {mask_dnt.sum()} cells')
print(f'  PBMC DNT: {(mask_dnt & (adata.obs["source"]=="PBMC")).sum()}')
print(f'  Tissue DNT: {(mask_dnt & (adata.obs["source"]=="tissue")).sum()}')

ct_colors = {
    'IL10+EOMES+ DNT': '#C97C6D',
    'Treg':           '#E6B56A',
    'CD4 T':          '#7FAFD4',
    'CD8 T':          '#6FA38E',
    'γδT':            '#9A8CC4',
    'NKT':            '#E8C97A',
    'MKI67+ T':       '#9FA8A3',
    'Other T':        '#D6D3CC'
}

# ============================================================
# Part 4: Compute Neighborhood Graph
# ============================================================
if 'neighbors' not in adata.uns:
    if 'X_scvi' in adata.obsm:
        print('Computing neighbors from X_scvi...')
        sc.pp.neighbors(adata, use_rep='X_scvi', n_neighbors=15)
    else:
        sc.pp.neighbors(adata, n_neighbors=15)

# ============================================================
# Part 5: PAGA Analysis - Major Cell Types
# ============================================================
print('=== PAGA: major_type ===')
adata.obs['major_type'] = adata.obs['major_type'].astype('category')
sc.tl.paga(adata, groups='major_type')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sc.pl.paga(adata, color='major_type', ax=axes[0], show=False,
           fontsize=10, node_size_scale=2, edge_width_scale=0.5,
           threshold=0.05)
axes[0].set_title('PAGA connectivity', fontsize=14)
sc.pl.umap(adata, color='major_type', ax=axes[1], show=False, size=15,
           palette=ct_colors, title='Major type')
sc.pl.umap(adata, color='source', ax=axes[2], show=False, size=15,
           palette={'PBMC': '#A8D4F0', 'tissue': '#F5A623'},
           title='Compartment')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'FigB1_paga_overview.pdf'),
            dpi=300, bbox_inches='tight')
plt.show()
print('Saved: Fig_paga_overview.pdf')

# ============================================================
# Part 6: PAGA Analysis - Cell Type x Compartment
# ============================================================
print('=== PAGA: celltype x compartment ===')
adata.obs['ct_source'] = (
    adata.obs['major_type'].astype(str) + '_' +
    adata.obs['source'].astype(str)
)
adata.obs['ct_source'] = adata.obs['ct_source'].astype('category')
sc.tl.paga(adata, groups='ct_source')

fig, ax = plt.subplots(figsize=(10, 8))
sc.pl.paga(adata, ax=ax, show=False, fontsize=8,
           node_size_scale=1.5, edge_width_scale=0.3,
           threshold=0.03)
ax.set_title('PAGA: celltype x compartment', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'FigB2_paga_ct_source.pdf'),
            dpi=300, bbox_inches='tight')
plt.show()
print('Saved: FigB2_paga_ct_source.pdf')

# ============================================================
# Part 7: DNT-Only Pseudotime Analysis (356 cells)
# ============================================================

# Extract DNT cells only
dnt_mask = adata.obs['major_type'] == 'IL10+EOMES+ DNT'
dnt = adata[dnt_mask].copy()
print(f'DNT cells: {dnt.n_obs}')
print(dnt.obs['source'].value_counts())

# Use scVI latent space
rep_key = 'X_scvi' if 'X_scvi' in dnt.obsm else 'X_pca'
sc.pp.neighbors(dnt, use_rep=rep_key, n_neighbors=15)
sc.tl.umap(dnt)
sc.tl.diffmap(dnt, n_comps=10)

# Root = PBMC DNT centroid
pbmc_mask = dnt.obs['source'] == 'PBMC'
pbmc_idx = np.where(pbmc_mask)[0]
centroid = dnt.obsm['X_diffmap'][pbmc_idx].mean(axis=0)
dists = np.linalg.norm(dnt.obsm['X_diffmap'][pbmc_idx] - centroid, axis=1)
dnt.uns['iroot'] = pbmc_idx[np.argmin(dists)]

sc.tl.dpt(dnt, n_dcs=10)
print(f'Pseudotime range: {dnt.obs["dpt_pseudotime"].min():.3f} – {dnt.obs["dpt_pseudotime"].max():.3f}')
print(f'PBMC median: {dnt.obs.loc[pbmc_mask, "dpt_pseudotime"].median():.3f}')
print(f'Tissue median: {dnt.obs.loc[~pbmc_mask, "dpt_pseudotime"].median():.3f}')

# ============================================================
# Part 8: Plot DNT-Only Pseudotime
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1. UMAP colored by pseudotime
sc.pl.umap(dnt, color='dpt_pseudotime', ax=axes[0], show=False,
           title='DNT Pseudotime\n(root = PBMC centroid)', frameon=True)

# 2. UMAP colored by compartment
sc.pl.umap(dnt, color='source', ax=axes[1], show=False,
           palette={'PBMC': '#A8D4F0', 'tissue': '#F5A623'},
           title='DNT by compartment', frameon=True)

# 3. Violin + strip: PBMC vs tissue pseudotime
pt_pbmc = dnt.obs.loc[dnt.obs['source'] == 'PBMC', 'dpt_pseudotime'].dropna()
pt_tissue = dnt.obs.loc[dnt.obs['source'] == 'tissue', 'dpt_pseudotime'].dropna()

parts = axes[2].violinplot([pt_pbmc, pt_tissue], positions=[0, 1], showmedians=False, showextrema=False)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(['#A8D4F0', '#F5A623'][i])
    pc.set_alpha(0.4)

axes[2].boxplot([pt_pbmc, pt_tissue], positions=[0, 1], widths=0.15,
                patch_artist=True, showfliers=False, zorder=3,
                boxprops=dict(facecolor='white', linewidth=1.5),
                medianprops=dict(color='black', linewidth=2),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5))

# Scatter overlay for tissue (few points, show them all)
axes[2].scatter(np.ones(len(pt_tissue)) + np.random.normal(0, 0.02, len(pt_tissue)),
                pt_tissue, c='#C87800', s=20, alpha=0.8, zorder=4, edgecolors='white', linewidths=0.3)

# Stats
stat, pval = stats.mannwhitneyu(pt_pbmc, pt_tissue, alternative='two-sided')
axes[2].text(0.5, 1.05, f'P = {pval:.2e}', ha='center', transform=axes[2].transAxes,
             fontsize=12, fontweight='bold')
axes[2].set_xticks([0, 1])
axes[2].set_xticklabels([f'PBMC\n(n={len(pt_pbmc)})', f'Tissue\n(n={len(pt_tissue)})'], fontsize=13)
axes[2].set_ylabel('Pseudotime', fontsize=13)
axes[2].set_title('DNT: PBMC vs Tissue', fontsize=14)
axes[2].spines['top'].set_visible(False)
axes[2].spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'FigB3_dnt_pseudotime.pdf'), dpi=300, bbox_inches='tight')
plt.show()

# Stats summary
print(f'\nPBMC DNT pseudotime: median={pt_pbmc.median():.3f}, mean={pt_pbmc.mean():.3f}')
print(f'Tissue DNT pseudotime: median={pt_tissue.median():.3f}, mean={pt_tissue.mean():.3f}')
print(f'Mann-Whitney U={stat:.0f}, P={pval:.2e}')

# ============================================================
# Part 9: Gene Expression Along Pseudotime
# ============================================================

highlight_genes = ['SELL', 'IL7R', 'LGALS1', 'HLA-DRA', 'IL10', 'CXCR4', 'EOMES', 'GZMK']
highlight_genes = [g for g in highlight_genes if g in dnt.var_names]

# Extract expression matrix
expr_mat = dnt.X.toarray() if sp.issparse(dnt.X) else dnt.X

n = len(highlight_genes)
cols = 4
rows = 4
fig, axes = plt.subplots(rows, cols, figsize=(10, 10), sharex=True)
axes = axes.flatten()

pt = dnt.obs['dpt_pseudotime'].values
sort_idx = np.argsort(pt)
pt_sorted = pt[sort_idx]
source_sorted = dnt.obs['source'].values[sort_idx]
window = max(10, len(pt_sorted) // 6)

for i, gene in enumerate(highlight_genes):
    ax = axes[i]
    g_idx = list(dnt.var_names).index(gene)
    expr = expr_mat[sort_idx, g_idx]
    expr_smooth = pd.Series(expr).rolling(window=window, min_periods=1, center=True).mean().values

    pbmc_pts = source_sorted != 'tissue'
    tissue_pts = source_sorted == 'tissue'
    ax.scatter(pt_sorted[pbmc_pts], expr[pbmc_pts], s=6, alpha=0.2, color='#A8D4F0', zorder=0)
    ax.scatter(pt_sorted[tissue_pts], expr[tissue_pts], s=30, alpha=0.9, color='#F5A623', zorder=2,
               edgecolors='white', linewidths=0.5)
    ax.plot(pt_sorted, expr_smooth, linewidth=2.5, color='#9370DB', zorder=3)

    ax.axvspan(0.3, 0.8, alpha=0.06, color='#F5A623')
    ax.set_title(gene, fontsize=20, fontweight='bold')
    ax.tick_params(axis='both', labelsize=16)
    if i >= cols:
        ax.set_xlabel('Pseudotime', fontsize=18)
    if i % cols == 0:
        ax.set_ylabel('Expression', fontsize=18)

for j in range(len(highlight_genes), len(axes)):
    axes[j].set_visible(False)

plt.suptitle('IL10⁺EOMES⁺ DNT: Gene Expression Along Pseudotime',
             fontsize=20, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'FigB9_gene_trends_final.pdf'), dpi=300, bbox_inches='tight')
plt.show()
