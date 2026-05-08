#!/usr/bin/env python3
"""
PBMC-Tissue T Cell Merge and Cross-Compartment Integration

This script merges PBMC T cells with intestinal tissue T cells (BGI platform) and performs:
- Tissue T cell loading and sample name harmonization
- PBMC T cell extraction with DNT labels from script 2
- Cross-compartment scVI integration (PBMC vs tissue)
- Full-gene recovery for downstream analysis beyond HVGs
- Normalization and export of merged object

Input:
  - Tissue T cells: BGI scRNA-seq h5ad
  - PBMC T cells: Full PBMC h5ad + DNT-labeled T cell h5ad from script 2
Output: Merged PBMC+tissue T cell h5ad with scVI embeddings
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import scvi
import anndata as ad
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

sc.settings.verbosity = 2
sc.settings.set_figure_params(dpi=100, frameon=False)

OUT_DIR = '/public/home/chenjiaminggroup/yinjinwen/DNT-scRNA/IBD_PBMC/01_processed'
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================================
# Part 1: Load Tissue T Cells
# ============================================================================

bgi_path = '/public/home/chenjiaminggroup/yinjinwen/DNT-scRNA/IBDbgi/00_data/01_proccessed/adata_sc.h5ad'
adata = sc.read_h5ad(bgi_path)

t_subtypes = [
    'CD4+Tcm-like', 'CD4+ Trm', 'CD4+ Tn', 'CD4+ Treg',
    'Th17-like', 'Tfh', 'Cycling CD4',
    'CD8+Trm-MAIT', 'CD8+ Tem', 'CD8+ Trm', 'CD8+ Tn',
    'CD8+MAIT', 'Epi-interactive CD8',
    'γδIEL', 'γδT',
    'IL10+ DNT',
]
t_subtypes = [s for s in t_subtypes if s in adata.obs['subtype_anno_2'].values]

tissue_t = adata[adata.obs['subtype_anno_2'].isin(t_subtypes)].copy()
tissue_t.obs['source'] = 'tissue'
tissue_t.obs['subtype_anno_2'] = tissue_t.obs['subtype_anno_2'].astype(str)
tissue_t.obs.loc[tissue_t.obs['subtype_anno_2'] == 'IL10+ DNT', 'subtype_anno_2'] = 'MHC-II+IL10+EOMES+ DNT'

print(f'Tissue T cells: {tissue_t.n_obs}')
print(f'Tissue DNT: {(tissue_t.obs["subtype_anno_2"] == "MHC-II+IL10+EOMES+ DNT").sum()}')
print(tissue_t.obs['subtype_anno_2'].value_counts())

tissue_sample_map = {
    'A0060A1':       'A060A',
    'A0070A1':       'A070A',
    'A0079A1':       'A079A',
    'A013B':         'A013B',
    'A043A':         'A043A',
    'B0161A1':       'B161A',
    'B0163A1':       'B163A',
    'B0189A1':       'B189A',
    'B0225A1':       'B225A',
    'B0230A1':       'B230A',
    'B217A':         'B217A',
    'L1EHH2200211':  'A032A',
    'L1EHH2200212':  'A031A',
    'L1EHH2200213':  'A030A',
    'L1EHH2300154':  'A029A',
    'L1EHH2400763':  'A028A',
    'L1EHH2901425':  'A037A',
    'L1EHH2901426':  'A038A',
    'L1EHI1300124':  'A040A',
}

tissue_t.obs['sample'] = tissue_t.obs['sample'].map(tissue_sample_map)
tissue_t.obs['group'] = tissue_t.obs['group'].map({
    'Responder': 'R',
    'Nonresponder': 'NR',
    'Untreated': 'UNT',
})

print('=== Tissue ===')
print(tissue_t.obs.groupby('sample')['group'].first().to_string())
print(f'\n{tissue_t.obs["group"].value_counts()}')

print('=== Tissue ===')
print(tissue_t.obs.groupby('group')['sample'].nunique())
print(tissue_t.obs['group'].value_counts())
print('\n--- Tissue samples ---')
for g in tissue_t.obs['group'].unique():
    samples = tissue_t.obs.loc[tissue_t.obs['group'] == g, 'sample'].unique()
    print(f'{g}: {samples}')

# ============================================================================
# Part 2: Load PBMC T Cells
# ============================================================================

PBMC_PATH = '/public/home/chenjiaminggroup/wufan/tempFiles/pbmcMerge.afterFindMarkers_RmBE_plusGroup_rmTHMTinHVG.20250801.h5ad'
pbmc_orig = sc.read_h5ad(PBMC_PATH)

pbmc_labeled = sc.read_h5ad('/public/home/chenjiaminggroup/yinjinwen/DNT-scRNA/IBD_PBMC/01_processed/pbmc_T_cells_DNT_labeled.h5ad')

common_bc = pbmc_orig.obs_names.intersection(pbmc_labeled.obs_names)
print(f'Common barcodes: {len(common_bc)} / {pbmc_labeled.n_obs}')

pbmc_t = pbmc_orig[common_bc].copy()
pbmc_t.X = pbmc_t.layers['counts'].copy()

pbmc_t.obs['cell_type_PBMC'] = pbmc_labeled.obs.loc[common_bc, 'cell_type_PBMC']
pbmc_t.obs['DNT_final'] = pbmc_labeled.obs.loc[common_bc, 'DNT_final']

pbmc_t.obs['source'] = 'PBMC'
pbmc_t.obs['subtype_anno_2'] = pbmc_t.obs['cell_type_PBMC'].astype(str)
pbmc_t.obs.loc[pbmc_t.obs['DNT_final'] == 'True', 'subtype_anno_2'] = 'MHC-II+IL10+EOMES+ DNT'

pbmc_t.obs['group'] = pbmc_t.obs['Group'].map({
    'R': 'R', 'NR': 'NR', 'UNT': 'UNT'
})

# Remove NK cells from PBMC T cell set
pbmc_remove = ['CD56bright NK', 'CD56dim NK']
pbmc_t = pbmc_t[~pbmc_t.obs['cell_type_PBMC'].isin(pbmc_remove)].copy()

_nz = pbmc_t.X[pbmc_t.X.nonzero()]
if hasattr(_nz, 'A1'): _nz = _nz.A1
print(f'PBMC X: min={_nz.min():.1f}, max={_nz.max():.1f}, int={np.allclose(_nz[:1000], _nz[:1000].astype(int))}')
print(f'PBMC T cells: {pbmc_t.n_obs}')
print(f'PBMC DNT: {(pbmc_t.obs["subtype_anno_2"] == "MHC-II+IL10+EOMES+ DNT").sum()}')

# Print sample-group summaries
print('\n=== PBMC ===')
print(pbmc_t.obs.groupby('group')['sample'].nunique())
print(pbmc_t.obs['group'].value_counts())
print('\n--- PBMC samples ---')
for g in pbmc_t.obs['group'].unique():
    samples = pbmc_t.obs.loc[pbmc_t.obs['group'] == g, 'sample'].unique()
    print(f'{g}: {samples}')

print('=== Tissue samples ===')
tissue_sg = tissue_t.obs.groupby('sample')['group'].first().reset_index()
tissue_sg.columns = ['sample', 'group']
print(tissue_sg.to_string(index=False))

print('\n=== PBMC samples ===')
pbmc_sg = pbmc_t.obs.groupby('sample')['group'].first().reset_index()
pbmc_sg.columns = ['sample', 'group']
print(pbmc_sg.to_string(index=False))

# Verify PBMC raw counts
print(f'PBMC raw: {pbmc_t.raw is not None}')
if pbmc_t.raw is not None:
    _nz = pbmc_t.raw.X[pbmc_t.raw.X.nonzero()]
    if hasattr(_nz, 'A1'): _nz = _nz.A1
    print(f'raw.X: min={_nz.min():.1f}, max={_nz.max():.1f}, int={np.allclose(_nz[:1000], _nz[:1000].astype(int))}')

_nz = pbmc_t.X[pbmc_t.X.nonzero()]
if hasattr(_nz, 'A1'): _nz = _nz.A1
print(f'X: min={_nz.min():.1f}, max={_nz.max():.1f}, int={np.allclose(_nz[:1000], _nz[:1000].astype(int))}')

# ============================================================================
# Part 3: Merge on Common Genes
# ============================================================================

common_genes = tissue_t.var_names.intersection(pbmc_t.var_names)
print(f'Common genes: {len(common_genes)}')

tissue_t = tissue_t[:, common_genes].copy()
pbmc_t = pbmc_t[:, common_genes].copy()

# Ensure raw counts in X
if 'counts' in tissue_t.layers:
    tissue_t.X = tissue_t.layers['counts'].copy()
_nz = tissue_t.X[tissue_t.X.nonzero()]
if hasattr(_nz, 'A1'): _nz = _nz.A1
print(f'Tissue X: min={_nz.min():.1f}, max={_nz.max():.1f}, int={np.allclose(_nz[:1000], _nz[:1000].astype(int))}')

if 'counts' in pbmc_t.layers:
    pbmc_t.X = pbmc_t.layers['counts'].copy()
_nz = pbmc_t.X[pbmc_t.X.nonzero()]
if hasattr(_nz, 'A1'): _nz = _nz.A1
print(f'PBMC X: min={_nz.min():.1f}, max={_nz.max():.1f}, int={np.allclose(_nz[:1000], _nz[:1000].astype(int))}')

# Harmonize obs columns
keep_cols = ['sample', 'group', 'source', 'subtype_anno_2']
for col in keep_cols:
    if col not in tissue_t.obs.columns:
        tissue_t.obs[col] = 'unknown'
    if col not in pbmc_t.obs.columns:
        if col == 'group' and 'Group' in pbmc_t.obs.columns:
            pbmc_t.obs['group'] = pbmc_t.obs['Group']
        else:
            pbmc_t.obs[col] = 'unknown'

tissue_t.obs = tissue_t.obs[keep_cols].copy()
pbmc_t.obs = pbmc_t.obs[keep_cols].copy()

combined = ad.concat([tissue_t, pbmc_t], merge='same')
combined.obs_names_make_unique()
print(f'\nCombined: {combined.n_obs} cells, {combined.n_vars} genes')
print(combined.obs['source'].value_counts())
print(f'DNT: {(combined.obs["subtype_anno_2"] == "MHC-II+IL10+EOMES+ DNT").sum()}')

combined.obs.loc[combined.obs['sample'] == 'B273AP', 'group'] = 'NR'

df_check = combined.obs.groupby('sample')['group'].first().reset_index()
print(df_check.to_string(index=False))

# ============================================================================
# Part 4: scVI Integration (PBMC + Tissue)
# ============================================================================

adata = combined.copy()
adata.obs['sample'] = adata.obs['sample'].astype(str)

sc.pp.filter_genes(adata, min_cells=10)

try:
    sc.pp.highly_variable_genes(
        adata, n_top_genes=3000, flavor='seurat_v3',
        batch_key='sample', subset=True, span=0.8
    )
except (ValueError, ZeroDivisionError):
    print('HVG batch fallback')
    sc.pp.highly_variable_genes(
        adata, n_top_genes=3000, flavor='seurat_v3',
        subset=True, span=1.0
    )
print(f'HVGs: {adata.n_vars}')

scvi.model.SCVI.setup_anndata(
    adata,
    batch_key='sample',
    categorical_covariate_keys=['source'],
)
vae = scvi.model.SCVI(adata, n_latent=30, n_layers=2, gene_likelihood='zinb')
vae.train(max_epochs=200, early_stopping=True)

adata.obsm['X_scvi'] = vae.get_latent_representation()

sc.pp.neighbors(adata, use_rep='X_scvi')
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=1.0, key_added='leiden_res1.0')

sc.pl.umap(adata, color=['source', 'leiden_res1.0'], ncols=2)
sc.pl.umap(adata, color=['subtype_anno_2'], legend_fontsize=6)
print(f'Done. {adata.n_obs} cells, {adata.n_vars} HVGs')

print(f'adata: {adata.shape}')
print(f'obsm keys: {list(adata.obsm.keys())}')
print(adata.obs['source'].value_counts())
print(adata.obs['subtype_anno_2'].value_counts())

adata.obs['is_DNT'] = adata.obs['subtype_anno_2'].str.contains('DNT').astype(str)
sc.pl.umap(adata, color='is_DNT', palette={'True': 'red', 'False': '#d3d3d3'}, size=10)

# DNT source labeling
adata.obs['DNT_source'] = 'other'
is_dnt = adata.obs['subtype_anno_2'].str.contains('DNT', na=False)
adata.obs.loc[is_dnt, 'DNT_source'] = adata.obs.loc[is_dnt, 'source']

sources = adata.obs.loc[is_dnt, 'source'].unique().tolist()
colors = plt.cm.tab10.colors
palette = {src: f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
           for src, (r, g, b, *_) in zip(sources, colors)}
palette['other'] = '#d3d3d3'

# ============================================================================
# Part 5: Full-Gene Recovery
# ============================================================================

tissue_full = sc.read_h5ad('/public/home/chenjiaminggroup/yinjinwen/DNT-scRNA/IBDbgi/00_data/01_proccessed/adata_sc.h5ad')
pbmc_full = sc.read_h5ad('/public/home/chenjiaminggroup/yinjinwen/DNT-scRNA/IBD_PBMC/01_processed/pbmc_T_cells_DNT_labeled.h5ad')

common_barcodes = adata.obs_names
tissue_full = tissue_full[tissue_full.obs_names.isin(common_barcodes)]
pbmc_full = pbmc_full[pbmc_full.obs_names.isin(common_barcodes)]

print(f'tissue: {tissue_full.n_obs}, pbmc: {pbmc_full.n_obs}')
print(f'total: {tissue_full.n_obs + pbmc_full.n_obs}, expected: {adata.n_obs}')

common_genes = tissue_full.var_names.intersection(pbmc_full.var_names)
print(f'Common genes (full): {len(common_genes)}')

full = ad.concat([tissue_full[:, common_genes], pbmc_full[:, common_genes]],
                 join='inner', merge='first')

full = full[adata.obs_names].copy()

# Transfer scVI embeddings and metadata
full.obsm['X_scvi'] = adata.obsm['X_scvi']
full.obsm['X_umap'] = adata.obsm['X_umap']
full.obsp = adata.obsp
full.obs = adata.obs.copy()

print(f'full: {full.shape}')
print(f'EOMES in var: {"EOMES" in full.var_names}')
print(f'CXCR4 in var: {"CXCR4" in full.var_names}')
print(f'S1PR4 in var: {"S1PR4" in full.var_names}')

# Normalize full-gene object
full.layers['counts'] = full.X.copy()
sc.pp.normalize_total(full, target_sum=1e4)
sc.pp.log1p(full)

print('Done. Use full for all downstream analysis.')

# ============================================================================
# Part 6: Save
# ============================================================================

out_path = os.path.join(OUT_DIR, 'pbmc_tissueBGI_merge_T_cells_DNT_labeled.h5ad')
adata.write(out_path)
print(f'Saved: {out_path}')
