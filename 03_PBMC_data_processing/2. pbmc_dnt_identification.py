#!/usr/bin/env python3
"""
PBMC T Cell Extraction and DNT Identification

This script extracts T cells from the full PBMC dataset and performs:
- T cell-specific scVI re-integration
- High-resolution clustering and cell type annotation
- DNT (Double-Negative T cell) identification based on signature genes
- Two-round annotation: first removes contamination, second identifies 17 T cell subtypes
- Final DNT cluster: MHC-II+IL10+EOMES+ DNT (cluster 15)

Input: Full PBMC h5ad from script 1
Output: T cell h5ad with DNT annotation
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import scvi
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.cm import Set3
import seaborn as sns

warnings.filterwarnings('ignore')

sc.settings.verbosity = 2
sc.settings.set_figure_params(dpi=100, frameon=False)

# DNT signature genes
dnt_sig3 = [
    'EOMES', 'LYST', 'GZMK', 'IL10',
    'TIGIT', 'CRTAM',
    'PECAM1', 'CD27', 'FOXP1', 'PHLDA1', 'CD44', 'CNR2', 'S1PR4', 'FABP5', 'ANXA2',
    'PIP4K2A', 'ACTG1', 'CLEC2D', 'AFF3', 'RUNX2', 'TCF7', 'PTPRJ', 'PIK3R5',
    'CD38', 'SLAMF7',
    'GPR183', 'CXCR4',
    'HLA-DRA', 'HLA-DRB1',
    'CD74',
    'MYB',
    'BACH2', 'CD81',
    'HAVCR2',
]

# Paths
PBMC_PATH = '/public/home/chenjiaminggroup/wufan/tempFiles/pbmcMerge.afterFindMarkers_RmBE_plusGroup_rmTHMTinHVG.20250801.h5ad'
OUT_DIR = '/public/home/chenjiaminggroup/yinjinwen/DNT-scRNA/IBD_PBMC/01_processed'
os.makedirs(OUT_DIR, exist_ok=True)

ref_dir = '/public/home/chenjiaminggroup/yinjinwen/DNT-scRNA/RA/models/ibd_t_scanvi_reference'
ibd_path = '/public/home/chenjiaminggroup/yinjinwen/IBD/202501scRNA/ALL20250111/T/annotation/T_group_scvi_umap_reso2_annotated.h5ad'

# ============================================================================
# Part 1: Load Full PBMC and Extract T Cells
# ============================================================================

pbmc = sc.read_h5ad(PBMC_PATH)

print(pbmc)
print(f'\nn_obs: {pbmc.n_obs}, n_vars: {pbmc.n_vars}')
print(f'X dtype: {pbmc.X.dtype}, X max: {pbmc.X.max():.2f}')
print(f'layers: {list(pbmc.layers.keys())}')
print(f'obsm: {list(pbmc.obsm.keys())}')
print(f'raw: {pbmc.raw is not None}')
print(f'\nobs columns:\n  {pbmc.obs.columns.tolist()}')

print(pbmc.obs['cell_type2'].value_counts())
print(pbmc.obs['cell_type3'].value_counts())
print(pbmc.obs['lineage'].value_counts())

# Inspect categorical metadata
for col in pbmc.obs.columns:
    if pbmc.obs[col].dtype == 'O' or str(pbmc.obs[col].dtype).startswith('category'):
        n_unique = pbmc.obs[col].nunique()
        if n_unique < 50:
            print(f'\n{col} ({n_unique} unique):')
            print(f'  {dict(pbmc.obs[col].value_counts().head(20))}')

# Check if X is raw counts or log-normalized
if sp.issparse(pbmc.X):
    nz = pbmc.X[pbmc.X.nonzero()].A1
else:
    nz = pbmc.X[pbmc.X != 0]
print(f'X nonzero min/max: {nz.min():.3f} / {nz.max():.3f}')
print(f'All integers? {np.all(nz[:1000] % 1 == 0)}')
print(f'Looks like: {"raw counts" if np.all(nz[:1000] % 1 == 0) else "log-normalized"}')

for layer in pbmc.layers.keys():
    L = pbmc.layers[layer]
    nzL = L[L.nonzero()].A1 if sp.issparse(L) else L[L != 0]
    print(f'\nlayer "{layer}": min/max = {nzL.min():.3f} / {nzL.max():.3f}, integers? {np.all(nzL[:1000] % 1 == 0)}')

if pbmc.raw is not None:
    rX = pbmc.raw.X
    nzR = rX[rX.nonzero()].A1 if sp.issparse(rX) else rX[rX != 0]
    print(f'\nraw.X: min/max = {nzR.min():.3f} / {nzR.max():.3f}, integers? {np.all(nzR[:1000] % 1 == 0)}')

# Existing UMAP / clusters overview
if 'X_umap' in pbmc.obsm:
    cluster_cols = [c for c in pbmc.obs.columns
                    if 'leiden' in c.lower() or 'cluster' in c.lower() or 'celltype' in c.lower()]
    print(f'Cluster-like columns: {cluster_cols}')
    show_cols = cluster_cols[:2]
    for c in ['group', 'disease']:
        if c in pbmc.obs.columns and c not in show_cols:
            show_cols.append(c)
    sc.pl.umap(pbmc, color=show_cols, ncols=2, legend_loc='on data', legend_fontsize=7)

# Extract T cells with IFX or untreated
CELLTYPE_COL = 'lineage'
T_CELL_LABELS = ['T']

mask_t = pbmc.obs[CELLTYPE_COL].isin(T_CELL_LABELS)
t = pbmc[mask_t].copy()
t = t[t.obs['Drug'].isin(['IFX', '/'])].copy()

print('Matched labels:')
print(t.obs[CELLTYPE_COL].value_counts())
print('\nDrug distribution:')
print(t.obs['Drug'].value_counts())
print(f'\nT cells: {t.n_obs} / {pbmc.n_obs} ({100*t.n_obs/pbmc.n_obs:.1f}%)')

# ============================================================================
# Part 2: scVI Re-integration
# ============================================================================

RAW_SOURCE = 'layers:counts'
if RAW_SOURCE.startswith('layers:'):
    layer = RAW_SOURCE.split(':')[1]
    t.X = t.layers[layer].copy()
    print(f'Using t.layers[{layer!r}] as raw counts')
elif RAW_SOURCE == 'raw':
    raw_ad = t.raw.to_adata()
    common = t.var_names.intersection(raw_ad.var_names)
    t2 = raw_ad[:, common].copy()
    t2.obs = t.obs.copy()
    for k in t.obsm.keys():
        t2.obsm[k] = t.obsm[k].copy()
    t = t2
    print(f'Pulled counts from t.raw ({t.n_vars} genes)')

_nz = t.X[t.X.nonzero()].A1 if sp.issparse(t.X) else t.X[t.X != 0]
print(f'After: X nonzero min/max = {_nz.min():.3f} / {_nz.max():.3f}, integers? {np.all(_nz[:1000] % 1 == 0)}')

# HVG on raw counts
sc.pp.filter_genes(t, min_cells=10)

BATCH_KEY = 'sample'
if BATCH_KEY not in t.obs.columns:
    print(f'WARNING: {BATCH_KEY} not in obs, using batch_key=None')
    BATCH_KEY = None

sc.pp.highly_variable_genes(t, flavor='seurat_v3', n_top_genes=3000,
                             batch_key=BATCH_KEY, layer='counts')
print(f'Genes after filter: {t.n_vars}')
print(f'HVG selected: {t.var.highly_variable.sum()}')

# scVI training
cat_covs = [c for c in ['sample', 'Group'] if c in t.obs.columns]
cont_covs = [c for c in ['pct_counts_mt'] if c in t.obs.columns]
print(f'Categorical covariates: {cat_covs}')
print(f'Continuous covariates: {cont_covs}')

scvi.model.SCVI.setup_anndata(
    t,
    layer='counts',
    batch_key=cat_covs[0] if cat_covs else None,
    categorical_covariate_keys=cat_covs[1:] if len(cat_covs) > 1 else None,
    continuous_covariate_keys=cont_covs or None,
)
vae = scvi.model.SCVI(t, n_latent=30, n_layers=2, gene_likelihood='zinb')
vae.train(max_epochs=200, early_stopping=True)
t.obsm['X_scvi'] = vae.get_latent_representation()

# UMAP + multi-resolution leiden
sc.pp.neighbors(t, use_rep='X_scvi')
sc.tl.umap(t)

for res in [0.5, 0.8, 1.0, 1.5, 2.0]:
    sc.tl.leiden(t, resolution=res, key_added=f'leiden_res{res}')
    n = t.obs[f'leiden_res{res}'].nunique()
    print(f'res={res}: {n} clusters')

fig_cols = [f'leiden_res{r}' for r in [0.5, 1.0, 1.5, 2.0]]
sc.pl.umap(t, color=fig_cols, ncols=2, legend_loc='on data', legend_fontsize=6)

# ============================================================================
# Part 3: First-Round Annotation and Contamination Removal
# ============================================================================

USE_RES = 'leiden_res1.5'

# Normalize for DEG / scoring / dotplot
t.layers['counts'] = t.X.copy()
sc.pp.normalize_total(t, target_sum=1e4)
sc.pp.log1p(t)
t.raw = t.copy()

sc.tl.dendrogram(t, groupby=USE_RES, use_rep='X_scvi')
sc.tl.rank_genes_groups(t, groupby=USE_RES, method='wilcoxon', use_raw=False)
sc.pl.rank_genes_groups(t, n_genes=10, sharey=False)

for cl in t.obs[USE_RES].cat.categories:
    df = sc.get.rank_genes_groups_df(t, group=cl).head(20)
    print(f'\n=== Cluster {cl} ===')
    print(df[['names', 'logfoldchanges', 'pvals_adj']].to_string(index=False))

# T subtype marker dotplot
t_markers = [
    'CD3E', 'CD3D', 'CD4', 'CD8A', 'CD8B',
    'FOXP3', 'IL2RA',
    'EOMES', 'IL10', 'GZMK', 'GPR183', 'HAVCR2', 'AFF3',
    'GZMB', 'PRF1', 'NKG7',
    'CCR7', 'SELL', 'TCF7', 'LEF1',
    'TRDC', 'TRGC1',
    'MKI67', 'TOP2A',
    'HLA-DRA', 'CD74', 'SLAMF7', 'CD27',
]
avail_t = [g for g in t_markers if g in t.var_names]
sc.pl.dotplot(t, var_names=avail_t, groupby=USE_RES,
              standard_scale='var', figsize=(13, 8))

# First-round annotation
cluster_annotation = {
    '0':  'CD8 Naive',
    '1':  'Ribosomal-high T',
    '2':  'CD4 Tcm',
    '3':  'CD8 CTL',
    '4':  'CD4 Naive',
    '5':  'NK',
    '6':  'GZMK+ CD8 Tem',
    '7':  'NK (CD56dim)',
    '8':  'gdT',
    '9':  'Treg',
    '10': 'CD4 memory',
    '11': 'Ribosomal-high T',
    '12': 'MAIT',
    '13': 'Proliferating T',
    '14': 'gdT (activated)',
    '15': 'Monocyte',
    '16': 'NKT',
    '17': 'NKT (small)',
    '18': 'IL10+ DNT',
    '19': 'Low-quality',
    '20': 'Erythroid',
    '21': 'Low-quality',
}

CONTAM_CLUSTERS = ['11', '15', '17', '19', '20', '21']

t.obs['cell_type_PBMC'] = t.obs[USE_RES].map(cluster_annotation).astype('category')
sc.pl.umap(t, color='cell_type_PBMC', legend_loc='right margin')
print(t.obs['cell_type_PBMC'].value_counts())

plot_markers = {
    'Identity':             ['CD3E', 'TRBC1', 'CD4', 'CD8A'],
    'Immune Suppression':   ['IL10', 'TIGIT', 'FOXP1', 'CD38'],
    'Effector/Activation':  ['GZMK', 'CRTAM', 'PHLDA1', 'SLAMF7', 'CD27', 'LYST'],
    'Migration':            ['PECAM1', 'CD44', 'CNR2', 'S1PR4', 'GPR183', 'CXCR4', 'CD81'],
    'Metabolism':           ['FABP5', 'ANXA2', 'PIP4K2A', 'ACTG1'],
    'Antigen Presentation': ['HLA-DRA', 'HLA-DRB1', 'CD74', 'CLEC2D'],
    'TF':                   ['EOMES', 'AFF3', 'MYB', 'RUNX2', 'TCF7', 'BACH2'],
    'Signal Regulation':    ['PTPRJ', 'PIK3R5'],
}
sc.pl.dotplot(t, var_names=plot_markers, groupby='cell_type_PBMC',
              standard_scale='var', figsize=(13, 8))

t = t[~t.obs[USE_RES].isin(CONTAM_CLUSTERS)].copy()
print(f'Clean T cells: {t.n_obs} (removed contamination)')

# ============================================================================
# Part 4: Second-Round Clustering and Final Annotation
# ============================================================================

sc.pp.neighbors(t, use_rep='X_scvi')
sc.tl.umap(t)

for res in [0.5, 1.0, 1.5]:
    sc.tl.leiden(t, resolution=res, key_added=f'leiden_res{res}')
    n = t.obs[f'leiden_res{res}'].nunique()
    print(f'res={res}: {n} clusters')

sc.pl.umap(t, color=[f'leiden_res{r}' for r in [0.5, 1.0, 1.5]],
           ncols=3, legend_loc='on data', legend_fontsize=6)

USE_RES = 'leiden_res1.5'
print(f'Clusters: {t.obs[USE_RES].nunique()}')
print(t.obs[USE_RES].value_counts().sort_index())

for cl in t.obs[USE_RES].cat.categories:
    df = sc.get.rank_genes_groups_df(t, group=cl).head(80)
    print(f'\n=== Cluster {cl} ===')
    print(df[['names', 'logfoldchanges', 'pvals_adj']].to_string(index=False))

sc.pl.dotplot(t, var_names=avail_t, groupby=USE_RES,
              standard_scale='var', figsize=(13, 10))

# Remove low-quality cluster
t = t[t.obs[USE_RES] != '17'].copy()

# Final annotation with DNT cluster
cluster_annotation = {
    '0':  'CD8 Tn',
    '1':  'CD4 Tn',
    '2':  'CD8 CTL',
    '3':  'CD4 Tcm',
    '4':  'TSHZ2+ CD4 Tcm',
    '5':  'CD56bright NK',
    '6':  'CD56dim NK',
    '7':  'GZMK+ CD8 Tem',
    '8':  'gdT',
    '9':  'Treg',
    '10': 'CD4 Tn',
    '11': 'CD4 Tmem',
    '12': 'Proliferating T',
    '13': 'MAIT',
    '14': 'Activated gdT',
    '15': 'MHC-II+IL10+EOMES+ DNT',
    '16': 'NKT',
}

t.obs['cell_type_PBMC'] = t.obs[USE_RES].map(cluster_annotation).astype('category')
print(t.obs['cell_type_PBMC'].value_counts())

t.obs['DNT_final'] = (t.obs[USE_RES] == '15').astype(str)
n_dnt = (t.obs['DNT_final'] == 'True').sum()
print(f'DNT_final: {n_dnt} cells')

avail = [g for g in dnt_sig3 if g in t.var_names]
sc.tl.score_genes(t, gene_list=avail, score_name='DNT_IBD_score')

mean_score = t.obs.groupby(USE_RES)['DNT_IBD_score'].mean().sort_values(ascending=False)
print('Mean DNT_IBD score per cluster:')
print(mean_score)

# ============================================================================
# Part 5: Visualization with Ordered Cell Types
# ============================================================================

cell_order = [
    'CD4 Tn', 'CD8 Tn',
    'CD4 Tcm', 'TSHZ2+ CD4 Tcm', 'CD4 Tmem',
    'GZMK+ CD8 Tem', 'CD8 CTL',
    'Treg',
    'Proliferating T',
    'MAIT', 'gdT', 'Activated gdT', 'NKT',
    'CD56bright NK', 'CD56dim NK',
    'MHC-II+IL10+EOMES+ DNT',
]

plot_markers = {
    'Identity':             ['CD3E', 'CD247', 'TRBC1', 'CD4', 'CD8A'],
    'Immune Suppression':   ['IL10', 'TIGIT', 'FOXP1', 'CD38'],
    'Effector/Activation':  ['GZMK', 'CRTAM', 'PHLDA1', 'SLAMF7', 'CD27', 'LYST'],
    'Migration':            ['PECAM1', 'CD44', 'CNR2', 'S1PR4', 'GPR183', 'CXCR4', 'CD81'],
    'Metabolism':           ['FABP5', 'ANXA2', 'PIP4K2A', 'ACTG1'],
    'Antigen Presentation': ['HLA-DRA', 'HLA-DRB1', 'CD74', 'CLEC2D'],
    'TF':                   ['EOMES', 'AFF3', 'MYB', 'RUNX2', 'TCF7', 'BACH2'],
    'Signal Regulation':    ['PTPRJ', 'PIK3R5'],
}

t.obs['cell_type_PBMC'] = pd.Categorical(
    t.obs['cell_type_PBMC'], categories=cell_order, ordered=True
)

sc.pl.matrixplot(t, var_names=plot_markers, groupby='cell_type_PBMC',
                 standard_scale='var', use_raw=False,
                 cmap='Reds', figsize=(14, 6), show=False)
plt.savefig('Fig_PBMC_heatmap_ordered.pdf', dpi=300, bbox_inches='tight')
plt.show()

sc.pl.dotplot(t, var_names=plot_markers, groupby='cell_type_PBMC',
              standard_scale='var', use_raw=False,
              figsize=(14, 6), show=False)
plt.savefig('Fig_PBMC_dotplot_ordered.pdf', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# Part 6: Final UMAP with DNT Highlighted
# ============================================================================

t = sc.read_h5ad('/public/home/chenjiaminggroup/yinjinwen/DNT-scRNA/IBD_PBMC/01_processed/pbmc_T_cells_DNT_labeled.h5ad')
clean = t.copy()

cats = cell_order
palette = {
    'CD4 Tn':              '#1f77b4',
    'CD8 Tn':              '#aec7e8',
    'CD4 Tcm':             '#2ca02c',
    'TSHZ2+ CD4 Tcm':      '#98df8a',
    'CD4 Tmem':            '#006400',
    'GZMK+ CD8 Tem':       '#ff7f0e',
    'CD8 CTL':             '#ffbb78',
    'Treg':                '#9467bd',
    'Proliferating T':     '#f7b6d2',
    'MAIT':                '#8c564b',
    'gdT':                 '#c49c94',
    'Activated gdT':       '#d62728',
    'NKT':                 '#bcbd22',
    'CD56bright NK':       '#17becf',
    'CD56dim NK':          '#7f7f7f',
    'MHC-II+IL10+EOMES+ DNT': '#d62728',
}

fig, ax = plt.subplots(figsize=(8.5, 4.5))
sc.pl.umap(clean, color='cell_type_PBMC',
           palette=palette,
           title='IBD PBMC T/NK cells', size=15, ax=ax, show=False)
plt.tight_layout()
plt.savefig('Fig_PBMC_panelA_landscape.pdf', dpi=300, bbox_inches='tight')
plt.show()

print("DNT identification complete.")
