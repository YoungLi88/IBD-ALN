"""
Cross-compartment analysis of DNT and T-cell subsets.

Generates:
  1. PBMC vs Tissue DNT transcriptome correlation scatter plot
  2. Migration receptor bubble dot-plot (heatmap)
  3. Radar plot of migration module scores

Input : pbmc_tissueBGI_merge_T_cells_DNT_labeled.h5ad
Output: figures and CSV tables in OUT_DIR
"""

import os
import warnings
from math import pi

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from numpy.polynomial.polynomial import polyfit

warnings.filterwarnings('ignore')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

sc.settings.verbosity = 2
sc.settings.set_figure_params(dpi=150, frameon=False, fontsize=12)

# ============================================================
# 1. Paths
# ============================================================
DATA_PATH = '/public/home/chenjiaminggroup/yinjinwen/DNT-scRNA/IBD_PBMC/01_processed/pbmc_tissueBGI_merge_T_cells_DNT_labeled.h5ad'
OUT_DIR = '/public/home/chenjiaminggroup/yinjinwen/DNT-scRNA/IBD_PBMC/02_cross_compartment'
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# 2. Load & prepare
# ============================================================
print('Loading data...')
adata = sc.read_h5ad(DATA_PATH)
print(f'Loaded: {adata.shape}')
print(f'obs columns: {adata.obs.columns.tolist()}')
print(f'layers: {list(adata.layers.keys())}')
print(f'obsm: {list(adata.obsm.keys())}')
print(adata.obs['source'].value_counts())

# Normalize if needed
if adata.X.max() > 20:
    print('Normalizing...')
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

# ============================================================
# 3. Define cell type groups (major_type based)
# ============================================================
if 'major_type' not in adata.obs.columns:
    subtype_to_major = {
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
    adata.obs['major_type'] = adata.obs['subtype_anno_2'].map(
        lambda x: subtype_to_major.get(x, 'Other T')
    )
    mask_dnt = adata.obs['is_DNT'] == True
    adata.obs.loc[mask_dnt, 'major_type'] = 'DNT'

print('major_type distribution:')
print(adata.obs['major_type'].value_counts())

# ============================================================
# 4. Migration Gene Module Scoring
# ============================================================
MODULES = {
    'Egress': [
        'S1PR1', 'S1PR4', 'KLF2', 'SELL', 'CCR7', 'TCF7',
        'LEF1', 'FOXP1', 'BACH2', 'ID3', 'KLF3', 'CXCR4', 'GPR183',
    ],
    'Adhesion': [
        'SELPLG', 'SELL', 'ITGAL', 'ITGB2', 'ITGA4', 'ITGB1',
        'ITGA4', 'ITGB7', 'ICAM3', 'CD2', 'ADAM17', 'FERMT3',
        'TLN1', 'RAP1A', 'VASP',
    ],
    'Diapedesis': [
        'CD99', 'PECAM1', 'CD44', 'JAM2', 'JAM3', 'RAC2', 'CDC42',
        'CORO1A', 'CORO2A', 'MMP9', 'MMP14', 'VAV1', 'MYH9',
        'RAPGEF1', 'ARHGAP15',
    ],
    'Retention': [
        'CD69', 'ITGAE', 'ITGA1', 'CXCR6', 'ZNF683', 'RGS1',
        'PRDM1', 'RUNX3', 'BHLHE40', 'NR4A1', 'AHR', 'RBPJ',
        'LITAF', 'FABP5',
    ],
}

for mod_name, genes in MODULES.items():
    avail = [g for g in genes if g in adata.var_names]
    missing = [g for g in genes if g not in adata.var_names]
    print(f'{mod_name}: {len(avail)}/{len(genes)} available, missing: {missing}')
    if len(avail) >= 2:
        sc.tl.score_genes(adata, gene_list=avail, score_name=f'{mod_name}_score')
    elif len(avail) == 1:
        idx = list(adata.var_names).index(avail[0])
        vals = adata.X[:, idx].toarray().ravel() if sp.issparse(adata.X) else adata.X[:, idx]
        adata.obs[f'{mod_name}_score'] = vals

score_cols = [f'{m}_score' for m in MODULES if f'{m}_score' in adata.obs.columns]
print(f'\nScored modules: {score_cols}')

# ============================================================
# 5. PBMC vs Tissue DNT transcriptome correlation scatter plot
# ============================================================

# 5.1 Compute per-gene mean expression for PBMC / Tissue DNT
mask_dnt = adata.obs['major_type'] == 'IL10+EOMES+ DNT'
dnt = adata[mask_dnt].copy()

pbmc_mask = dnt.obs['source'] == 'PBMC'
tissue_mask = dnt.obs['source'] == 'tissue'

pbmc_mean = (np.array(dnt[pbmc_mask].X.mean(axis=0)).ravel()
             if sp.issparse(dnt.X) else dnt[pbmc_mask].X.mean(axis=0))
tissue_mean = (np.array(dnt[tissue_mask].X.mean(axis=0)).ravel()
               if sp.issparse(dnt.X) else dnt[tissue_mask].X.mean(axis=0))

gene_corr = pd.DataFrame({
    'gene': dnt.var_names,
    'PBMC': pbmc_mean,
    'Tissue': tissue_mean,
})
gene_corr['diff'] = gene_corr['Tissue'] - gene_corr['PBMC']

# Pearson & Spearman
r_p, p_p = pearsonr(gene_corr['PBMC'], gene_corr['Tissue'])
r_s, p_s = spearmanr(gene_corr['PBMC'], gene_corr['Tissue'])

# Regression line
b, m = polyfit(gene_corr['PBMC'], gene_corr['Tissue'], 1)

fig, ax = plt.subplots(figsize=(5, 5))

ax.scatter(gene_corr['PBMC'], gene_corr['Tissue'],
           s=6, alpha=0.3, c='#999', rasterized=True)

# Regression line
x_line = np.linspace(0, gene_corr['PBMC'].max() * 1.05, 100)
ax.plot(x_line, b + m * x_line, color='#E63946', linewidth=2,
        label=f'slope = {m:.3f}')

# y=x reference line
ax.plot(x_line, x_line, color='grey', linestyle='--', linewidth=1,
        alpha=0.5, label='y = x')

# Annotate correlation coefficient
ax.text(0.05, 0.95, f'Pearson r = {r_p:.4f}',
        transform=ax.transAxes, fontsize=16, va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.set_xlabel('Mean Expression (PBMC DNT)', fontsize=16)
ax.set_ylabel('Mean Expression (Tissue DNT)', fontsize=16)
ax.set_title('Transcriptome Correlation: PBMC vs Tissue DNT', fontsize=15)
max_val = max(gene_corr['PBMC'].max(), gene_corr['Tissue'].max()) * 1.05
ax.set_xlim(0, 3)
ax.set_ylim(0, 3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'Fig_correlation_pbmc_tissue_DNT.pdf'),
            dpi=300, bbox_inches='tight')
plt.show()

# Export CSV
gene_corr.to_csv(os.path.join(OUT_DIR, 'correlation_pbmc_tissue_DNT.csv'),
                 index=False)

# 5.2 Compute module scores per group
celltypes = ['IL10+EOMES+ DNT', 'Treg', 'CD4 T', 'CD8 T', 'γδT']
compartments = ['PBMC', 'tissue']

records = []
for ct in celltypes:
    for comp in compartments:
        mask = (adata.obs['major_type'] == ct) & (adata.obs['source'] == comp)
        n = mask.sum()
        if n < 5:
            continue
        row = {'celltype': ct, 'compartment': comp, 'n_cells': n}
        for sc_name in score_cols:
            row[sc_name] = adata.obs.loc[mask, sc_name].mean()
        records.append(row)

df_scores = pd.DataFrame(records)
df_scores.to_csv(os.path.join(OUT_DIR, 'migration_module_scores.csv'),
                 index=False)
print(df_scores.to_string(index=False))

# ============================================================
# 6. Migration receptors bubble dot-plot
# ============================================================
migration_receptors = {
    # Chemokine receptors
    'CCR1': 'Chemokine', 'CCR2': 'Chemokine', 'CCR3': 'Chemokine',
    'CCR4': 'Chemokine', 'CCR5': 'Chemokine', 'CCR6': 'Chemokine',
    'CCR7': 'Chemokine', 'CCR8': 'Chemokine', 'CCR9': 'Chemokine',
    'CCR10': 'Chemokine',
    'CXCR1': 'Chemokine', 'CXCR2': 'Chemokine', 'CXCR3': 'Chemokine',
    'CXCR4': 'Chemokine', 'CXCR5': 'Chemokine', 'CXCR6': 'Chemokine',
    'CX3CR1': 'Chemokine', 'XCR1': 'Chemokine',
    # S1P receptors
    'S1PR1': 'S1P', 'S1PR2': 'S1P', 'S1PR3': 'S1P',
    'S1PR4': 'S1P', 'S1PR5': 'S1P',
    # Integrins
    'ITGAL': 'Integrin', 'ITGB2': 'Integrin',
    'ITGA4': 'Integrin', 'ITGB7': 'Integrin',
    'ITGB1': 'Integrin', 'ITGAE': 'Integrin', 'ITGA1': 'Integrin',
    # Adhesion / transendothelial migration
    'PECAM1': 'Adhesion', 'CD99': 'Adhesion', 'SELL': 'Adhesion',
    'SELPLG': 'Adhesion', 'CD44': 'Adhesion',
    # Lipid / metabolic migration
    'GPR183': 'Lipid', 'PTGER2': 'Lipid', 'PTGER4': 'Lipid',
    # Tissue residency / transcription factors
    'CD69': 'Residency', 'KLF2': 'TF', 'ZNF683': 'TF',
    'PRDM1': 'TF', 'RUNX3': 'TF',
}

key_migration_genes = [g for g in migration_receptors if g in adata.var_names]
avail_key = key_migration_genes
print(f'{len(key_migration_genes)}/{len(migration_receptors)} migration '
      f'receptors available\n')

celltypes = ['IL10+EOMES+ DNT', 'Treg', 'CD4 T', 'CD8 T', 'γδT']
compartments = ['PBMC', 'tissue']

slope_records = []
for ct in celltypes:
    for gene in avail_key:
        g_idx = list(adata.var_names).index(gene)
        for comp in ['PBMC', 'tissue']:
            mask = ((adata.obs['major_type'] == ct)
                    & (adata.obs['source'] == comp))
            if mask.sum() < 5:
                continue
            vals = adata.X[mask, g_idx]
            if sp.issparse(vals):
                vals = vals.toarray()
            slope_records.append({
                'gene': gene,
                'category': migration_receptors[gene],
                'celltype': ct,
                'compartment': comp,
                'mean_expr': vals.mean(),
            })

df_slope = pd.DataFrame(slope_records)
df_slope.to_csv(os.path.join(OUT_DIR, 'migration_receptors_expression.csv'),
                index=False)

df_wide = df_slope.pivot_table(
    index=['gene', 'category', 'celltype'],
    columns='compartment', values='mean_expr',
).reset_index()
df_wide['delta'] = df_wide['tissue'] - df_wide['PBMC']
print(f'Saved: migration_receptors_expression.csv\nShape: {df_wide.shape}')

# Custom colour map for dot-plot
custom_cmap = LinearSegmentedColormap.from_list(
    'pbmc_tissue', ['#6FA8DC', 'white', '#E6954A'])

celltypes_order = ['IL10+EOMES+ DNT', 'Treg', 'CD4 T', 'CD8 T', 'γδT']
ct_display = {
    'IL10+EOMES+ DNT': 'DNT', 'Treg': 'Treg',
    'CD4 T': 'CD4 T', 'CD8 T': 'CD8 T', 'γδT': 'γδT',
}
dnt_delta = (df_wide[df_wide['celltype'] == 'IL10+EOMES+ DNT']
             .set_index('gene')['delta'].to_dict())

cat_order_df = pd.DataFrame([
    {'gene': g, 'category': migration_receptors[g],
     'delta': dnt_delta.get(g, 0)}
    for g in avail_key
])
cat_mean = (cat_order_df.groupby('category')['delta']
            .mean().sort_values(ascending=False))
cat_order = cat_mean.index.tolist()

gene_order, gene_cat_list = [], []
for cat in cat_order:
    cat_genes = (cat_order_df[cat_order_df['category'] == cat]
                 .sort_values('delta', ascending=False)['gene'].tolist())
    gene_order.extend(cat_genes)
    gene_cat_list.extend([cat] * len(cat_genes))

gap = 0.5
y_positions, y, prev_cat = [], 0, None
cat_y_ranges = {}
for i, gene in enumerate(gene_order):
    cat = gene_cat_list[i]
    if prev_cat is not None and cat != prev_cat:
        y += gap
        cat_y_ranges[prev_cat] = (cat_y_ranges[prev_cat][0],
                                  y_positions[-1])
    if cat not in cat_y_ranges:
        cat_y_ranges[cat] = (y, y)
    y_positions.append(y)
    y += 1
    prev_cat = cat
cat_y_ranges[prev_cat] = (cat_y_ranges[prev_cat][0], y_positions[-1])
gene_to_y = {g: yp for g, yp in zip(gene_order, y_positions)}

group_order = ([f'{ct}_PBMC' for ct in celltypes_order]
               + [f'{ct}_tissue' for ct in celltypes_order])
adata.obs['ct_comp'] = (adata.obs['major_type'].astype(str) + '_'
                        + adata.obs['source'].astype(str))

records = []
for grp in group_order:
    mask = adata.obs['ct_comp'] == grp
    if mask.sum() < 5:
        continue
    sub = adata[mask]
    for gene in gene_order:
        if gene not in adata.var_names:
            continue
        g_idx = list(adata.var_names).index(gene)
        vals = sub.X[:, g_idx]
        if sp.issparse(vals):
            vals = vals.toarray().ravel()
        else:
            vals = vals.ravel()
        records.append({
            'group': grp, 'gene': gene,
            'mean_expr': vals.mean(),
            'pct_expr': (vals > 0).mean() * 100,
        })

dot_df = pd.DataFrame(records)
grp_to_x = {g: i for i, g in enumerate(group_order)}
dot_df['y'] = dot_df['gene'].map(gene_to_y)
dot_df['x'] = dot_df['group'].map(grp_to_x)

highlight = ['CXCR4', 'CCR7']
cat_cmap = {
    'Chemokine': '#D98C5F', 'S1P': '#8E8CC3', 'Integrin': '#6FA8DC',
    'Adhesion': '#7FB8A4', 'Lipid': '#E6C86B', 'Residency': '#7BAE7F',
    'TF': '#C97C6D',
}

size_min, size_max = 5, 120
expr_vals = dot_df['mean_expr']
dot_df['size'] = size_min + (expr_vals / expr_vals.max()) * (size_max - size_min)
dot_df['zscore'] = dot_df.groupby('gene')['mean_expr'].transform(
    lambda x: (x - x.mean()) / (x.std() + 1e-8))

fig = plt.figure(figsize=(5.5, 7.5))
gs = gridspec.GridSpec(1, 2, width_ratios=[0.3, 10], wspace=0.005)
ax_bar = fig.add_subplot(gs[0])
ax_main = fig.add_subplot(gs[1])

# Category colour bar (left panel)
for i, gene in enumerate(gene_order):
    cat = gene_cat_list[i]
    yp = y_positions[i]
    ax_bar.barh(yp, 1, color=cat_cmap.get(cat, '#ccc'), height=1,
                edgecolor=cat_cmap.get(cat, '#ccc'), linewidth=0)

ax_bar.set_ylim(min(y_positions) - 0.5, max(y_positions) + 0.5)
ax_bar.set_yticks(y_positions)
ax_bar.set_yticklabels(gene_order, fontsize=12)
ax_bar.invert_yaxis()
ax_bar.set_xticks([])
ax_bar.set_xlim(0, 1)
for spine in ax_bar.spines.values():
    spine.set_visible(False)
ax_bar.tick_params(left=False, pad=1)
for i, label in enumerate(ax_bar.get_yticklabels()):
    if gene_order[i] in highlight:
        label.set_fontweight('bold')

# Dot-plot (right panel)
ax_main.set_facecolor('white')
ax_main.grid(False)

for j in range(len(group_order) + 1):
    ax_main.axvline(j - 0.5, color='#eee', linewidth=0.4, zorder=1)
for yp in y_positions:
    ax_main.axhline(yp - 0.5, color='#eee', linewidth=0.2, zorder=1)
ax_main.axhline(y_positions[-1] + 0.5, color='#eee', linewidth=0.2,
                zorder=1)

sc_plot = ax_main.scatter(
    dot_df['x'], dot_df['y'],
    c=dot_df['zscore'], s=dot_df['size'],
    cmap=custom_cmap, vmin=-2, vmax=2,
    edgecolors='none', linewidths=0, zorder=3)

ax_main.set_xticks(range(len(group_order)))
xlabels = [ct_display.get(g.rsplit('_', 1)[0], g.rsplit('_', 1)[0])
           for g in group_order]
ax_main.set_xticklabels(xlabels, fontsize=12, rotation=60, ha='left')
ax_main.xaxis.set_ticks_position('top')
ax_main.xaxis.set_label_position('top')

mid = len(celltypes_order) - 0.5
ax_main.axvline(mid, color='black', linewidth=1, zorder=2)
ax_main.text(mid / 2, min(y_positions) - 1.8, 'PBMC',
             ha='center', fontsize=13, fontweight='bold', color='#579FD7')
ax_main.text(mid + len(celltypes_order) / 2, min(y_positions) - 1.8,
             'Tissue', ha='center', fontsize=13, fontweight='bold',
             color='#C87800')

ax_main.set_yticks(y_positions)
ax_main.set_yticklabels([])
ax_main.set_ylim(min(y_positions) - 0.5, max(y_positions) + 0.5)
ax_main.invert_yaxis()
for spine in ax_main.spines.values():
    spine.set_visible(False)
ax_main.tick_params(left=False, bottom=False, pad=1)

cbar = plt.colorbar(sc_plot, ax=ax_main, shrink=0.15, pad=0.02, aspect=10)
cbar.set_label('Z-score', fontsize=12)
cbar.ax.tick_params(labelsize=10)
cbar.outline.set_visible(False)

# Size legend
expr_max = dot_df['mean_expr'].max()
for val in [0.5, 1.0, 2.0]:
    if val <= expr_max:
        s = size_min + (val / expr_max) * (size_max - size_min)
        ax_main.scatter([], [], s=s, c='#ddd', edgecolors='#999',
                        linewidths=0.3, label=f'{val:.1f}')
size_leg = ax_main.legend(
    title='Mean Expr', loc='lower right', fontsize=10,
    title_fontsize=10, bbox_to_anchor=(1.25, 0.0),
    frameon=True, fancybox=False, edgecolor='#ccc',
    handletextpad=0.3, borderpad=0.3)
ax_main.add_artist(size_leg)

# Category legend
cat_patches = [Patch(facecolor=cat_cmap[c], edgecolor='white', label=c)
               for c in cat_order if c in cat_cmap]
cat_leg = ax_main.legend(
    handles=cat_patches, title='Category', loc='lower right', fontsize=10,
    title_fontsize=10, bbox_to_anchor=(1.25, 0.20),
    frameon=True, fancybox=False, edgecolor='#ccc',
    handletextpad=0.3, borderpad=0.3)
ax_main.add_artist(cat_leg)

plt.subplots_adjust(left=0.25, right=0.78, top=0.88, bottom=0.02)
plt.savefig(os.path.join(OUT_DIR, 'FigA_bubble_dotplot_receptors.pdf'),
            dpi=300, bbox_inches='tight')
plt.show()

# ============================================================
# 7. Radar plot for migration module scores
# ============================================================
module_names = [m.replace('_score', '') for m in score_cols]
angles = [n / float(len(module_names)) * 2 * pi
          for n in range(len(module_names))]
angles += [angles[0]]

cmap = plt.cm.get_cmap('tab10', len(celltypes))
colors_ct = {ct: cmap(i) for i, ct in enumerate(celltypes)}

fig, axes = plt.subplots(1, 2, figsize=(8, 5),
                         subplot_kw=dict(polar=True))
celltypes_plot = [ct for ct in celltypes if ct != 'γδT']
for ax, comp in zip(axes, compartments):
    ax.set_title(comp, fontsize=14, fontweight='bold', pad=20)
    for ct in celltypes_plot:
        row = df_scores[(df_scores['celltype'] == ct)
                        & (df_scores['compartment'] == comp)]
        if row.empty:
            continue
        values = ([row.iloc[0][sc_] for sc_ in score_cols]
                  + [row.iloc[0][score_cols[0]]])
        ax.plot(angles, values, 'o-', linewidth=2, color=colors_ct[ct],
                label=f'{ct} (n={int(row.iloc[0]["n_cells"])})')
        ax.fill(angles, values, alpha=0.15, color=colors_ct[ct])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(module_names, fontsize=12)
    ax.legend(loc='lower right', bbox_to_anchor=(1.0, 1.1), fontsize=9)

plt.suptitle('Migration Module Scores: PBMC vs Tissue', fontsize=16, y=1.05)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'FigA1_radar_migration_modules.pdf'),
            dpi=300, bbox_inches='tight')
plt.show()
