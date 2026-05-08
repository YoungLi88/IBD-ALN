#!/usr/bin/env python3
"""
Cross-Compartment UMAP Visualization (Figure 3 Panels)

This script generates a three-panel UMAP figure for the merged PBMC+tissue T cell dataset:
- Panel A: T cell major type annotation
- Panel B: Compartment of origin (PBMC vs tissue)
- Panel C: DNT co-localization overlay (PBMC DNT + tissue DNT)

Input: Merged PBMC+tissue T cell h5ad from script 3
Output: Three-panel UMAP figure (PDF)
"""

import os
import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import scvi
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings('ignore')

sc.settings.set_figure_params(dpi=150, frameon=False)

# ============================================================================
# Part 1: Paths and Data Loading
# ============================================================================

DATA_PATH = '/public/home/chenjiaminggroup/yinjinwen/DNT-scRNA/IBD_PBMC/01_processed/pbmc_tissueBGI_merge_T_cells_DNT_labeled.h5ad'
OUT_DIR = '/public/home/chenjiaminggroup/yinjinwen/DNT-scRNA/IBD_PBMC/02_cross_compartment'
os.makedirs(OUT_DIR, exist_ok=True)

adata = sc.read_h5ad(DATA_PATH)
print(f'Loaded: {adata.shape}')
print(f'obs columns: {adata.obs.columns.tolist()}')
print(f'layers: {list(adata.layers.keys())}')
print(f'obsm: {list(adata.obsm.keys())}')
print(adata.obs['source'].value_counts())

# ============================================================================
# Part 2: Normalize (preserve raw counts in layers)
# ============================================================================

_nz = adata.X[adata.X.nonzero()].A1 if sp.issparse(adata.X) else adata.X[adata.X != 0]
print(f'X: min={_nz.min():.3f}, max={_nz.max():.3f}, integers={np.all(_nz[:1000] % 1 == 0)}')

if np.all(_nz[:1000] % 1 == 0):
    print('X is raw counts, normalizing...')
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
else:
    print('X already log-normalized')

# ============================================================================
# Part 3: DNT Labeling and Major Type Consolidation
# ============================================================================

# DNT annotation
adata.obs['is_DNT'] = adata.obs['subtype_anno_2'].str.contains('DNT').astype(str)
mask_dnt = adata.obs['subtype_anno_2'].str.contains('DNT')
adata.obs['DNT_source'] = 'Other'
adata.obs.loc[mask_dnt & (adata.obs['source'] == 'tissue'), 'DNT_source'] = 'Tissue DNT'
adata.obs.loc[mask_dnt & (adata.obs['source'] == 'PBMC'), 'DNT_source'] = 'PBMC DNT'
print(f'\nDNT: {mask_dnt.sum()} cells')
print(adata.obs['DNT_source'].value_counts())

# Consolidate subtypes to major types
major_type_map = {
    # CD4
    'CD4 Tcm': 'CD4 T', 'CD4 Tmem': 'CD4 T', 'CD4 Tn': 'CD4 T',
    'CD4+ Tn': 'CD4 T', 'CD4+ Trm': 'CD4 T', 'CD4+Tcm-like': 'CD4 T',
    'TSHZ2+ CD4 Tcm': 'CD4 T', 'Cycling CD4': 'CD4 T',
    'CD4+ Treg': 'Treg', 'Treg': 'Treg',
    'Tfh': 'CD4 T', 'Th17-like': 'CD4 T',
    # CD8
    'CD8 Tn': 'CD8 T', 'CD8 CTL': 'CD8 T', 'CD8+ Tem': 'CD8 T',
    'CD8+ Tn': 'CD8 T', 'CD8+ Trm': 'CD8 T', 'CD8+MAIT': 'CD8 T',
    'CD8+Trm-MAIT': 'CD8 T', 'GZMK+ CD8 Tem': 'CD8 T',
    'Epi-interactive CD8': 'CD8 T', 'MAIT': 'CD8 T',
    # gdT / NKT
    'gdT': 'γδT', 'Activated gdT': 'γδT', 'NKT': 'NKT',
    'γδIEL': 'γδT', 'γδT': 'γδT',
    # Other
    'Proliferating T': 'MKI67+ T',
    # DNT
    'MHC-II+IL10+EOMES+ DNT': 'IL10+EOMES+ DNT',
}

# Check for unmapped subtypes
all_types = adata.obs['subtype_anno_2'].unique()
missing = [t for t in all_types if t not in major_type_map]
if missing:
    print(f'Unmapped types: {missing}')
    for t in missing:
        major_type_map[t] = 'Other T'

adata.obs['major_type'] = adata.obs['subtype_anno_2'].map(major_type_map)
print(adata.obs['major_type'].value_counts())

# ============================================================================
# Part 4: Three-Panel UMAP Figure
# ============================================================================

major_order = ['CD4 T', 'CD8 T', 'Treg', 'γδT', 'NKT', 'MKI67+ T', 'IL10+EOMES+ DNT']
adata.obs['major_type'] = pd.Categorical(adata.obs['major_type'], categories=major_order, ordered=True)

major_palette = {
    'CD4 T':           '#88C8E8',
    'CD8 T':           '#98CC78',
    'Treg':            '#F0B86A',
    'γδT':             '#B8A0D0',
    'NKT':             '#78C8B8',
    'MKI67+ T':        '#E8D888',
    'IL10+EOMES+ DNT': '#d62728',
}

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
coords = adata.obsm['X_umap']

# Panel A: Major types
sc.pl.umap(adata, color='major_type', ax=axes[0], show=False,
           palette=major_palette, title='T cell major types', size=12)

# Panel B: Compartment
sc.pl.umap(adata, color='source', ax=axes[1], show=False,
           palette={'PBMC': '#A8D4F0', 'tissue': '#F5A623'},
           title='Compartment', size=12)

# Panel C: DNT overlay
axes[2].scatter(coords[:, 0], coords[:, 1], s=1, c='#d3d3d3', alpha=0.3, rasterized=True)

pbmc_dnt_mask = mask_dnt & (adata.obs['source'] == 'PBMC')
axes[2].scatter(coords[pbmc_dnt_mask, 0], coords[pbmc_dnt_mask, 1],
                s=15, c='#A8D4F0', label=f'PBMC DNT (n={pbmc_dnt_mask.sum()})', zorder=3)

tissue_dnt_mask = mask_dnt & (adata.obs['source'] == 'tissue')
axes[2].scatter(coords[tissue_dnt_mask, 0], coords[tissue_dnt_mask, 1],
                s=60, c='#F5A623', edgecolors='#C87800', linewidths=0.5,
                label=f'Tissue DNT (n={tissue_dnt_mask.sum()})', zorder=4)

axes[2].legend(fontsize=10, loc='best', frameon=True)
axes[2].set_title('DNT co-localization')
axes[2].set_xlabel('UMAP1')
axes[2].set_ylabel('UMAP2')
axes[2].set_xticks([])
axes[2].set_yticks([])

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'Fig3_UMAP_panels_ABC.pdf'), dpi=300, bbox_inches='tight')
plt.show()
