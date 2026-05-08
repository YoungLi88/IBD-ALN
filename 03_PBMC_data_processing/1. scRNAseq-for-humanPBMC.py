#!/usr/bin/env python3
"""
PBMC scRNA-seq Analysis Pipeline

This script performs comprehensive single-cell RNA-seq analysis of human PBMC samples
from IBD patients treated with Infliximab (IFX). The pipeline includes:
- Quality control and filtering
- Doublet detection with Scrublet
- Batch effect correction using scVI
- Cell type annotation
- Differential expression analysis across treatment groups

Treatment groups: UNT (untreated), NR (non-responder), SOR (slow/other responder), R (responder)
Platform: 10x Genomics
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import anndata as ad
import matplotlib.pyplot as plt

sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=150, facecolor="white")

results_file = "/public/home/chenjiaminggroup/wufan/20250207_scRNAseq_IBD_PBMC_WF/analysis_res/write/"

# ============================================================================
# 1. Data Loading and Merging
# ============================================================================

sampleName = pd.read_table("/public/home/chenjiaminggroup/wufan/20250207_scRNAseq_IBD_PBMC_WF/analysis_res/sampleName.txt", header=None)
samples = sampleName[0].to_list()
adatas = {}

for sample_id in samples:
    path = "/public/home/chenjiaminggroup/wufan/20250207_scRNAseq_IBD_PBMC_WF/"+sample_id+"_scRNA/outs/filtered_feature_bc_matrix.h5"
    sample_adata = sc.read_10x_h5(path)
    sample_adata.var_names_make_unique()
    adatas[sample_id] = sample_adata

adata = ad.concat(adatas, label="sample")
adata.obs_names_make_unique()
print(adata)

# ============================================================================
# 2. Quality Control Metrics
# ============================================================================

adata.var["mt"] = adata.var_names.str.startswith("MT-")
adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]")

sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True)

mito_filter = 10
n_counts_filter = 5000
fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt', ax=axs[0], show=False)
sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts', ax=axs[1], show=False)
axs[0].hlines(y=mito_filter, xmin=0, xmax=max(adata.obs['total_counts']), color='red', ls='dashed', linewidth=0.5)
axs[1].hlines(y=n_counts_filter, xmin=0, xmax=max(adata.obs['total_counts']), color='red', ls='dashed', linewidth=0.5)
axs[1].hlines(y=400, xmin=0, xmax=max(adata.obs['total_counts']), color='red', ls='dashed', linewidth=0.5)
axs[1].vlines(x=25000, ymin=0, ymax=max(adata.obs['n_genes_by_counts']), color='red', ls='dashed', linewidth=0.5)
fig.tight_layout()
plt.savefig('QC.scatter_plots.png', dpi=150)
plt.close(fig)

sc.pl.violin(adata, ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
    stripplot=False, multi_panel=True, show=False, save="QC.violin.png")

sc.pl.scatter(adata, "total_counts", "n_genes_by_counts", color="pct_counts_mt",
    show=False, save="QC.scatter.png")

# ============================================================================
# 3. Cell and Gene Filtering
# ============================================================================

n0 = adata.shape[0]
print(f'Original cell number: {n0}\n')

sc.pp.filter_cells(adata, max_genes=5000)
n1 = adata.shape[0]
print(f'Higher threshold, n_genes_by_counts: 5000; filtered-out-cells: {n0-n1}, remain {n1} cells')

sc.pp.filter_cells(adata, min_genes=400)
n2 = adata.shape[0]
print(f'Lower threshold, n_genes_by_counts: 400; filtered-out-cells: {n1-n2}, remain {n2} cells')

adata = adata[adata.obs['pct_counts_mt'] < 15]
n3 = adata.shape[0]
print(f'Higher threshold, pct_counts_mt: 15%; filtered-out-cells: {n2-n3}, remain {n3} cells')

adata = adata[adata.obs['total_counts'] <= 25000]
n4 = adata.shape[0]
print(f'Removing the outlier cells in scatter plot: {n3-n4}, remain {n4} cells, last cells before rm doublet\n')

g0 = adata.shape[1]
sc.pp.filter_genes(adata, min_cells=3)
print(f'Gene threshold, min_cells: 3; filtered-out-genes: {g0-adata.shape[1]}, remain {adata.shape[1]} genes')

sc.pl.violin(adata, ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
    stripplot=False, multi_panel=True, show=False, save="QC.violin.afterFilter.png")

sc.pl.scatter(adata, "total_counts", "n_genes_by_counts", color="pct_counts_mt",
    show=False, save="QC.scatter.afterFilter.png")

# ============================================================================
# 4. Doublet Detection
# ============================================================================

sc.pp.scrublet(adata, batch_key="sample")
print(adata.obs.groupby("sample")["predicted_doublet"].value_counts().unstack(fill_value=0))
print('\n')

adata = adata[~adata.obs['predicted_doublet']].copy()
n5 = adata.shape[0]
print(f'Removing the doublet cells by scrublet: {n4-n5}, remain {n5} cells, last cells after rm doublet\n')

sc.pl.scrublet_score_distribution(adata, scale_hist_obs='log', scale_hist_sim='log', show=False, save="QC.doublets.png")

# ============================================================================
# 5. Normalization and Metadata Integration
# ============================================================================

adata.layers["counts"] = adata.X.copy()

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata

sampleInfor = pd.read_csv('/public/home/chenjiaminggroup/wufan/20250207_scRNAseq_IBD_PBMC_WF/analysis_res/sampleInformation_clear.txt', sep='\t')
merge_df = adata.obs.merge(sampleInfor, on='sample', how='left').copy()
merge_df.index = adata.obs.index
adata.obs = merge_df
del(merge_df)
del(sampleInfor)

adata.obs.drop(columns=['Description', 'predicted_doublet', 'Lib_time', 'Tissue', 'Type', 'IBD'], inplace=True)
adata = adata[adata.obs['Group'] != '_']

mt_genes = adata.var[adata.var['mt']].index.to_list()
ribo_genes = adata.var[adata.var['ribo']].index.to_list()
hb_genes = adata.var[adata.var['hb']].index.to_list()
adata = adata[:, ~adata.var.index.isin(mt_genes + ribo_genes + hb_genes)].copy()

# ============================================================================
# 6. Batch Effect Correction with scVI
# ============================================================================

sc.pp.highly_variable_genes(
    adata, n_top_genes=2000, subset=True,
    layer="counts", flavor="seurat_v3", batch_key="sample")

scvi.model.SCVI.setup_anndata(adata, layer="counts",
    categorical_covariate_keys=["sample", "Group"],
    continuous_covariate_keys=["pct_counts_mt"])

num_layers = 2
num_latent = 30
disp = "gene-batch"
g_like = "zinb"
model = scvi.model.SCVI(adata, n_layers=num_layers, n_latent=num_latent, gene_likelihood=g_like, dispersion=disp)

print("scVI model info: ", model, '\n')

model.train(max_epochs=1000, early_stopping=True)

model_dir = os.path.join(results_file, "scvi_model_plusGroup_rmTHMTinHVG"+'_'+str(num_layers)+'_'+str(num_latent)+'_'+disp+'_'+g_like)
model.save(model_dir, overwrite=True)

train_elbo = model.history['elbo_train'][1:]
test_elbo = model.history['elbo_validation']
ax = train_elbo.plot()
test_elbo.plot(ax=ax).figure.savefig("model_scvi_plusGroup_rmTHMTinHVG_loss.png")

# ============================================================================
# 7. Dimensionality Reduction and Clustering
# ============================================================================

SCVI_LATENT_KEY = "X_scVI"
latent = model.get_latent_representation()
adata.obsm[SCVI_LATENT_KEY] = latent
print(latent.shape)

sc.pp.neighbors(adata, use_rep='X_scVI', n_neighbors=30, knn=True)
sc.tl.umap(adata, min_dist=0.5)
sc.pl.umap(adata, color=["sample", "Group", "Sex", "Drug", "pct_counts_mt", "pct_counts_ribo"],
    size=2, show=False, wspace=0.5, hspace=0.5, ncols=3,
    save="UMAP.NNG.plot.scVI_RmBEs_plusGroup_rmTHMTinHVG_neighbors30.png")

for res in [0.15, 0.5, 0.18, 0.2, 0.22]:
    sc.tl.leiden(adata, key_added=f"leiden_res_{res:4.2f}", resolution=res, n_iterations=2, flavor="igraph")
    sc.pl.umap(adata, color=[f"leiden_res_{res:4.2f}"], show=False, legend_loc='on data',
        save=f"leiden_res_{res:4.2f}"+"_clustering.RmBE_plusGroup_rmTHMTinHVG.png")

adata.write(results_file + 'pbmcMerge.afterNNG.scVI_RmBE_plusGroup_rmTHMTinHVG.20250214.h5ad')

# ============================================================================
# 8. Marker Gene Identification
# ============================================================================

for res in [0.15, 0.18, 0.22, 0.2]:
    sc.tl.rank_genes_groups(adata, groupby=f"leiden_res_{res:4.2f}", method="wilcoxon", use_raw=True)
    sc.pl.rank_genes_groups_dotplot(adata, groupby=f"leiden_res_{res:4.2f}", standard_scale="var", n_genes=5,
        show=False, save=f"dotplot_markers.leiden_res_{res:4.2f}.RmBE_plusGroup_rmTHMTinHVG.png")
    print("Marker genes:")
    print(pd.DataFrame(adata.uns["rank_genes_groups"]["names"]).head(5))
    print("\n")
    markersDF1 = sc.get.rank_genes_groups_df(adata, group=None, pval_cutoff=0.001, log2fc_min=1)
    markersDF1.to_csv(f"rank_genes_groups_{res:4.2f}.marker_genes_RmBE_plusGroup_rmTHMTinHVG.csv", index=False)

adata.obs.groupby('sample')['leiden_res_0.22'].value_counts().unstack(fill_value=0)

# ============================================================================
# 9. Cell Type Annotation
# ============================================================================

marker_genes = {
    "Tcell": ["CD3E", "CD3D", "IL7R"],
    'CD4T': ["CD4", "CD40LG"],
    "NKT": ["NKG7", "GZMA", "GNLY", "SYNE1", "GZMH", "GZMK"],
    "CD8T": ["CD8A", "CD8B", "NELL2"],
    "B": ["CD79A", "BANK1", "MS4A1", "IGHD", "FCRL1", "IGHM", "PAX5"],
    "plasmaB": ["JCHAIN", 'IRF4', "XBP1", "LILRA4"],
    "Mono": ["LYZ", "CD14", "FCN1", "CST3", "DMXL2"],
    "DC": ["CD83", 'HLA-DPA1', 'HLA-DPB1', 'HLA-DRA'],
    'pDC': ['IRF8', 'TCF4', 'UGCG', 'PTPRS'],
    "Precursor": ["CD34", "KIT", 'CDK6'],
    "Mega(platelets)": ["PPBP", "PF4", "NRGN", "GP1BB"],
    "Neutrophil": ['CSF3R', "FCGR3B", "NAMPT", "NEAT1", "AQP9", "G0S2"],
    'Basophil': ['ENPP3', 'CCR3', 'IL3RA'],
}

sc.pl.dotplot(adata, marker_genes, groupby="leiden_res_0.20", standard_scale="var", use_raw=True,
    show=False, save="250401.dotplot_markers.leiden_res_0.20.RmBE_plusGroup_rmTHMTinHVG_markerGenes.png")

# Remove contaminating clusters
my_list = adata.obs[adata.obs['leiden_res_0.50'].isin(['5', '9', '16']) |
                    adata.obs['leiden_res_0.22'].isin(['2', '7'])].index.to_list()
rmCells = pd.Series(my_list).unique().tolist()
adata = adata[~adata.obs.index.isin(rmCells)].copy()
adata.write(results_file+'pbmcMerge.afterFindMarkers_RmBE_plusGroup_rmTHMTinHVG.20250214.celltype.rm259.h5ad')

adata.obs["cell_type2"] = adata.obs["leiden_res_0.20"].map({
    "0": "Neutrophil",
    "1": "plasmaB",
    "2": "NKT",
    "3": "Precursor",
    "4": "T(DNT+CD4T)",
    "5": "CD8T",
    "6": "Mega(Platelets)",
    "7": "Mono",
    "8": "DC",
    "9": "B",
    "10": "Basophil",
    "11": "pDC",
})

adata.obs["leiden_res_0.22_new"] = adata.obs["leiden_res_0.20"].map({
    "0": "10",
    "1": "4",
    "2": "1",
    "3": "8",
    "4": "0",
    "5": "2",
    "6": "9",
    "7": "5",
    "8": "6",
    "9": "3",
    "10": "11",
    "11": "7",
})

desired_order = ["T(DNT+CD4T)", "NKT", "CD8T", "B", "plasmaB", "Mono", "DC", "pDC", "Precursor", "Mega(Platelets)", "Neutrophil", "Basophil"]
adata.obs['cell_type2'] = pd.Categorical(adata.obs['cell_type2'], categories=desired_order, ordered=True)

desired_order = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
adata.obs['leiden_res_0.22_new'] = pd.Categorical(adata.obs['leiden_res_0.22_new'], categories=desired_order, ordered=True)

adata.uns['leiden_res_0.22_new_colors'] = ['#1f77b4', '#17becf', '#279e68', '#e377c2', '#aa40fc', '#8c564b', '#ff7f0e', '#b5bd61', '#d62728', '#aec7e8', '#ffbb78', '#98df8a']
adata.uns['cell_type2_colors'] = ['#1f77b4', '#17becf', '#279e68', '#e377c2', '#aa40fc', '#8c564b', '#ff7f0e', '#b5bd61', '#d62728', '#aec7e8', '#ffbb78', '#98df8a']

sc.pl.umap(adata, color=['leiden_res_0.22_new', 'cell_type2'],
    frameon=False, legend_loc='right margin',
    show=False, save="250401.UMAP.plot.scVI_RmBEs_plusGroup_rmTHMTinHVG_celltype1.2.png")

sc.pl.umap(adata, color=['leiden_res_0.22_new', 'cell_type2'],
    frameon=False, legend_loc='on data',
    show=False, save="250401.UMAP.plot.scVI_RmBEs_plusGroup_rmTHMTinHVG_celltype1.2.2.png")

sc.pl.umap(adata, color=['leiden_res_0.22_new', 'cell_type2'],
    frameon=False, legend_loc='right margin', size=2, add_outline=True,
    show=False, save="250401.UMAP.plot.scVI_RmBEs_plusGroup_rmTHMTinHVG_celltype1.2.1.png")

sc.pl.dotplot(adata, marker_genes, groupby="leiden_res_0.22_new", standard_scale="var", use_raw=True,
    show=False, save="250401.dotplot_markers.leiden_res_0.20.RmBE_plusGroup_rmTHMTinHVG_markerGenes.png")

fig = sc.pl.dotplot(adata, marker_genes, groupby="leiden_res_0.22_new", standard_scale="var", use_raw=True,
    return_fig=False, show=False)
ax = fig["mainplot_ax"]
for l in ax.get_xticklabels():
    l.set_style("italic")
plt.savefig("250401.dotplot_markers.leiden_res_0.20.RmBE_plusGroup_rmTHMTinHVG_markerGenes.1.png",
    dpi=150, bbox_inches="tight")
plt.close()

# ============================================================================
# 10. Group-wise UMAP Visualization
# ============================================================================

ncols = 4
nrows = 1
figsize = 4
fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
    figsize=(ncols * figsize + figsize * 0.1 * (ncols-1), nrows * figsize))
plt.subplots_adjust(wspace=0.1)
sc.pl.umap(adata[adata.obs.Group.isin(["UNT"]), :], color='cell_type2', ax=axs[0], title='UNT', frameon=False, legend_loc=None, show=False)
sc.pl.umap(adata[adata.obs.Group.isin(["NR"]), :], color='cell_type2', ax=axs[1], title='NR', frameon=False, legend_loc=None, show=False)
sc.pl.umap(adata[adata.obs.Group.isin(["SOR"]), :], color='cell_type2', ax=axs[2], title='SOR', frameon=False, legend_loc=None, show=False)
sc.pl.umap(adata[adata.obs.Group.isin(["R"]), :], color='cell_type2', ax=axs[3], title='R', frameon=False, legend_loc='right margin', show=False)
plt.savefig("UMAP.plot.scVI_RmBEs_plusGroup_rmTHMTinHVG_celltype2.splitGroup.png", dpi=150, bbox_inches="tight")
plt.close()

adata.obs["lineage"] = adata.obs["leiden_res_0.22_new"].map({
    "0": "T", "1": "T", "2": "T", "3": "B", "4": "plasmaB", "5": "Mono",
    "6": "Mono", "7": "Mono", "8": "Precursor", "9": "Mega", "10": "Granu", "11": "Granu"
})

fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
    figsize=(ncols * figsize + figsize * 0.1 * (ncols-1), nrows * figsize))
plt.subplots_adjust(wspace=0.1)
sc.pl.umap(adata[adata.obs.Group.isin(["UNT"]), :], color='lineage', ax=axs[0], title='UNT', frameon=False, legend_loc=None, show=False)
sc.pl.umap(adata[adata.obs.Group.isin(["NR"]), :], color='lineage', ax=axs[1], title='NR', frameon=False, legend_loc=None, show=False)
sc.pl.umap(adata[adata.obs.Group.isin(["SOR"]), :], color='lineage', ax=axs[2], title='SOR', frameon=False, legend_loc=None, show=False)
sc.pl.umap(adata[adata.obs.Group.isin(["R"]), :], color='lineage', ax=axs[3], title='R', frameon=False, legend_loc='right margin', show=False)
plt.savefig("UMAP.plot.scVI_RmBEs_plusGroup_rmTHMTinHVG_lineage.splitGroup.png", dpi=150, bbox_inches="tight")
plt.close()

# ============================================================================
# 11. Cluster-wise Sub-analysis
# ============================================================================

for i in adata.obs['leiden_res_0.22_new'].unique():
    tempAdata = adata[adata.obs['leiden_res_0.22_new'].isin([str(i)]), :].copy()
    sc.pp.neighbors(tempAdata, use_rep='X_scVI', n_neighbors=30, knn=True)
    sc.tl.umap(tempAdata, min_dist=0.5)
    sc.tl.leiden(tempAdata, key_added=f"C{i}_leiden_res_0.22", resolution=0.22, n_iterations=2, flavor="igraph")
    sc.pl.umap(tempAdata, color=[f"C{i}_leiden_res_0.22"], show=False, legend_loc='on data',
        save=f"250401.UMAP_C{i}_leiden_res_0.22_clustering.RmBE_plusGroup_rmTHMTinHVG.cycle4.png")
    sc.tl.rank_genes_groups(tempAdata, groupby=f"C{i}_leiden_res_0.22", method="wilcoxon", use_raw=True)
    sc.pl.rank_genes_groups_dotplot(tempAdata, groupby=f"C{i}_leiden_res_0.22", standard_scale="var", n_genes=5,
        show=False, save=f"250401.dotplot_markers.C{i}_leiden_res_0.22.RmBE_plusGroup_rmTHMTinHVG.cycle4.png")
    markersDF1 = sc.get.rank_genes_groups_df(tempAdata, group=None, pval_cutoff=0.001, log2fc_min=1)
    markersDF1.to_csv(f"250401.rank_genes_groups_C{i}_0.22.marker_genes_RmBE_plusGroup_rmTHMTinHVG.cycle4.csv", index=False)
    tempAdata.raw = None
    tempAdata.write(results_file+f'250401.C{i}_pbmcMerge.afterFindMarkers_RmBE_plusGroup_rmTHMTinHVG.20250401.cycle4.h5ad')

# ============================================================================
# 12. Differential Expression Analysis by Treatment Group
# ============================================================================

adata.obs['Group2'] = adata.obs['Group']
adata.obs['Group2'] = adata.obs['Group2'].cat.add_categories(['NR+SOR'])
adata.obs.loc[adata.obs['Group'].isin(['NR', 'SOR']), 'Group2'] = 'NR+SOR'
adata.obs['Group2'] = pd.Categorical(adata.obs['Group2'], categories=["UNT", "NR+SOR", "R"], ordered=True)
adata.obs['Group3'] = adata.obs.apply(lambda row: f"{row['Group2']}_{row['cell_type2']}", axis=1)
adata.raw = adata.copy()

for i in adata.obs['cell_type2'].unique().tolist():
    a = 'UNT_'+i
    b = 'NR+SOR_'+i
    c = 'R_'+i

    sc.tl.rank_genes_groups(adata, groupby="Group3", groups=[a], reference=b, method="wilcoxon", use_raw=True)
    markersDF1 = sc.get.rank_genes_groups_df(adata, group=None, pval_cutoff=0.01)
    markersDF1.to_csv(f"250401.rank_genes_groups_{a}_vs_{b}_Group3_RmBE_plusGroup_rmTHMTinHVG.csv", index=False)

    sc.tl.rank_genes_groups(adata, groupby="Group3", groups=[a], reference=c, method="wilcoxon", use_raw=True)
    markersDF1 = sc.get.rank_genes_groups_df(adata, group=None, pval_cutoff=0.01)
    markersDF1.to_csv(f"250401.rank_genes_groups_{a}_vs_{c}_Group3_RmBE_plusGroup_rmTHMTinHVG.csv", index=False)

    sc.tl.rank_genes_groups(adata, groupby="Group3", groups=[b], reference=c, method="wilcoxon", use_raw=True)
    markersDF1 = sc.get.rank_genes_groups_df(adata, group=None, pval_cutoff=0.01)
    markersDF1.to_csv(f"250401.rank_genes_groups_{b}_vs_{c}_Group3_RmBE_plusGroup_rmTHMTinHVG.csv", index=False)

# ============================================================================
# 13. Final Data Preparation and Export
# ============================================================================

tt = adata.obs
tt['group_old'] = tt['Group']
tt['Group'] = tt['Group'].replace('SOR', 'NR')
tt['Group'] = pd.Categorical(tt['Group'], categories=['UNT', 'NR', 'R'], ordered=True)
adata.obs = tt

tt['cell_type2'] = tt['cell_type2'].cat.add_categories(['DNT'])
tt.loc[tt['cell_type3'] == '0_DNT', 'cell_type2'] = 'DNT'
tt['cell_type2'] = tt['cell_type2'].replace('T(DNT+CD4T)', 'CD4T')
tt['cell_type2'] = pd.Categorical(tt['cell_type2'],
    categories=['DNT', 'CD4T', 'CD8T', 'NKT', 'B', 'plasmaB', 'Mono',
        'DC', 'pDC', 'Precursor', 'Mega(Platelets)', 'Neutrophil', 'Basophil'],
    ordered=True)
adata.obs = tt

adata_sub = adata[(adata.obs['cell_type3'] != '1_Neu-del') & (adata.obs['cell_type3'] != '7_doublet-del')].copy()
adata_sub.raw = None
adata_sub.write(results_file + 'pbmcMerge.afterFindMarkers_RmBE_plusGroup_rmTHMTinHVG.20250801.h5ad')

print("Analysis complete.")
