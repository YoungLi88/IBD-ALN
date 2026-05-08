#!~/miniconda3/envs/scvi-env/bin/python

import scanpy as sc
import anndata as ad
import pandas as pd
import matplotlib.pyplot as plt

sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=300, facecolor="white")

results_file = "/public/home/chenjiaminggroup/wufan/20251111_scRNAseq_MouseModel_YJW/write/"

sampleName = pd.read_table("/public/home/chenjiaminggroup/wufan/20251111_scRNAseq_MouseModel_YJW/sampleName.txt",header=None)
samples = sampleName[0].to_list()
adatas = {}

## Read samples and merge all samples
for sample_id in samples:
    path = "/public/home/chenjiaminggroup/publicdata/20251111_scRNAseq_MouseModel_YJW/matrix/"+sample_id+"_filtered_feature_bc_matrix.h5"
    sample_adata = sc.read_10x_h5(path)
    sample_adata.var_names_make_unique()
    adatas[sample_id] = sample_adata

adata = ad.concat(adatas, label="sample")
adata.obs_names_make_unique()
print(adata.obs["sample"].value_counts())
print(adata)

temp_geneID = sample_adata.var
adata.var = temp_geneID.loc[adata.var.index]
del temp_geneID


## Filter cells and genes
# mitochondrial genes, "MT-" for human, "Mt-" for mouse
adata.var["mt"] = adata.var_names.str.startswith(("mt-", "Mt-"))
adata.var["ribo"] = adata.var_names.str.startswith(("Rps", "Rpl"))
adata.var["hb"] = adata.var_names.str.contains("^Hb[^(p)]")
adata.var.loc[adata.var.index.isin(['Hbs1l','Hbegf']),'hb']=False

sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True)
# percent_top=[20]

mito_filter = 20
n_counts_filter = 7000
fig, axs = plt.subplots(ncols = 2, figsize = (8,4))
sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt',ax = axs[0], show=False)
sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts',ax = axs[1], show = False)
axs[0].hlines(y = mito_filter, xmin = 0, xmax = max(adata.obs['total_counts']), color = 'red', ls = 'dashed',linewidth=0.5)
axs[1].hlines(y = n_counts_filter, xmin = 0, xmax = max(adata.obs['total_counts']), color = 'red', ls = 'dashed',linewidth=0.5)
axs[1].hlines(y = 500, xmin = 0, xmax = max(adata.obs['total_counts']), color = 'red', ls = 'dashed',linewidth=0.5)
axs[1].vlines(x = 40000, ymin = 0, ymax = max(adata.obs['n_genes_by_counts']), color = 'red', ls = 'dashed',linewidth=0.5)
fig.tight_layout()
plt.savefig('QC.scatter_plots.png', dpi=150)
plt.close(fig)

sc.pl.violin(adata, ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
    stripplot = False, # jitter=0.4,
    multi_panel=True, show=False, 
    save="QC.violin.png")

sc.pl.scatter(adata, "total_counts", "n_genes_by_counts", color="pct_counts_mt",
    show=False,
    save="QC.scatter.png")


n0 = adata.shape[0]
print(f'Original cell number: {n0}')
print("\n")


sc.pp.filter_cells(adata, max_genes=7000)
n1 = adata.shape[0]
print(f'Higher treshold, n_genes_by_counts: 7000; filtered-out-cells: {n0-n1}, remain {n1} cells')

sc.pp.filter_cells(adata, min_genes=500)
n2 = adata.shape[0]
print(f'Lower treshold, n_genes_by_counts: 500; filtered-out-cells: {n1-n2}, remain {n2} cells')

adata = adata[adata.obs['pct_counts_mt']<20]
n3 = adata.shape[0]
print(f'Higher treshold, pct_counts_mt: 10%; filtered-out-cells: {n2-n3}, remain {n3} cells')


adata = adata[adata.obs['total_counts']<=40000]
n4 = adata.shape[0]
print(f'Removing the outlier cells in scatter plot: {n3-n4}, remain {n4} cells, last cells before rm doublet')
print("\n")


g0 = adata.shape[1]
sc.pp.filter_genes(adata, min_cells=3)
print(f'Gene treshold, min_cells: 3; filtered-out-genes: {g0-adata.shape[1]}, remain {adata.shape[1]} genes')


## Doublet detection
sc.pp.scrublet(adata, batch_key="sample")

print(adata.obs.groupby("sample")["predicted_doublet"].value_counts().unstack(fill_value=0))
print('\n')

adata = adata[~adata.obs['predicted_doublet']].copy()
n5 = adata.shape[0]
print(f'Removing the doublet cells by scrublet: {n4-n5}, remain {n5} cells, last cells after rm doublet')
print("\n")

sc.pl.scrublet_score_distribution(adata, scale_hist_obs = 'log', 
    scale_hist_sim = 'log', show=False, save="QC.doublets.png")

## add the group information
adata.obs.loc[adata.obs['sample'].isin(['LC_2','LC_6','LC_7','LC_8','LC_9','LC_12']),'Group'] = 'TKO'
adata.obs.loc[adata.obs['sample'].isin(['LC_1','LC_3','LC_4','LC_5','LC_10','LC_11']),'Group'] = 'WT'

adata.write(results_file + 'scRNAseq_Mm_TKOIL10_processing1.h5ad')


### Remove batch effect
adata = sc.read_h5ad(results_file+"scRNAseq_Mm_TKOIL10_processing1.h5ad")
adata.layers["counts"] = adata.X.copy()

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata) #log(normalize+1)
adata.raw = adata

mt_genes = adata.var[adata.var['mt']].index.to_list()
ribo_genes = adata.var[adata.var['ribo']].index.to_list()
hb_genes = adata.var[adata.var['hb']].index.to_list()
adata = adata[:,~adata.var.index.isin(mt_genes + ribo_genes + hb_genes)].copy()

### remove batch effect by scVI
sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True,
    layer="counts", flavor="seurat_v3", batch_key="sample")

scvi.model.SCVI.setup_anndata(adata, layer="counts", 
    categorical_covariate_keys=["sample"],
    continuous_covariate_keys=["pct_counts_mt"])

num_layers = 2
num_latent = 30
disp="gene-batch"
g_like = "zinb"
model = scvi.model.SCVI(adata, n_layers=num_layers, 
    n_latent=num_latent, gene_likelihood=g_like,
    dispersion=disp)

print("scVI model infor: ", model,'\n')

model.train(max_epochs=1000, early_stopping=True)

model_dir = os.path.join(results_file, "scRNAseq_Mm_TKOIL10_processing2"+'_'+str(num_layers)+'_'+str(num_latent)+'_'+disp+'_'+g_like)
model.save(model_dir, overwrite=True)

train_elbo = model.history['elbo_train'][1:]
test_elbo = model.history['elbo_validation']
ax = train_elbo.plot()
test_elbo.plot(ax=ax).figure.savefig("model_scvi_scRNAseq_processing2_loss.png")


### Obtaining model outputs
SCVI_LATENT_KEY = "X_scVI"
latent = model.get_latent_representation()
adata.obsm[SCVI_LATENT_KEY] = latent
print(latent.shape)

# Nearest neighbor graph constuction and visualization
adata_test = adata.copy()
for n_n in [15,20,25,30]: # original is 15
    sc.pp.neighbors(adata_test, use_rep='X_scVI', n_neighbors = n_n, knn=True)
    sc.tl.umap(adata_test, min_dist = 0.3) 
    sc.pl.umap(adata_test, color=["sample", "Group", "pct_counts_mt","pct_counts_ribo"],
        size=2, show=False, wspace=0.5,hspace=0.5,ncols=3,
        save='UMAP.plot.scVI_'+f"neighbors{n_n}"+'testNeighbors.png')
del(adata_test)

sc.pp.neighbors(adata, use_rep='X_scVI', n_neighbors = 30, knn=True)
sc.tl.umap(adata, min_dist = 0.5) 
sc.pl.umap(adata, color=["sample", "Group","pct_counts_mt","pct_counts_ribo"],
    size=2, show=False, wspace=0.5,hspace=0.5,ncols=3,
    save="UMAP.plot_neighbors30.png")

for res in [0.15, 0.18, 0.2, 0.25, 0.3]:
    sc.tl.leiden(adata, key_added=f"leiden_res_{res:4.2f}", resolution=res, n_iterations = 2, flavor="igraph")
    sc.pl.umap(adata, color=[f"leiden_res_{res:4.2f}"], show=False, legend_loc = 'on data',
        save=f"leiden_res_{res:4.2f}"+"_clustering.testRes.UMAP.png")

# sc.tl.leiden(adata, key_added="leiden_res_0.30", resolution=0.3, n_iterations = 2, flavor="igraph")
# sc.pl.umap(adata, color=["leiden_res_0.30"], show=False, legend_loc = 'on data',
#     save="leiden_res_0.30_clustering.UMAP.png")


for res in [0.3]:
    sc.tl.rank_genes_groups(adata, groupby=f"leiden_res_{res:4.2f}", 
        method="wilcoxon", use_raw = True)
    sc.pl.rank_genes_groups_dotplot(adata, groupby=f"leiden_res_{res:4.2f}", 
        standard_scale="var", n_genes=5,
        show=False, save=f"dotplot_markers.leiden_res_{res:4.2f}.cluster.markerGenes.png")
    markersDF1 = sc.get.rank_genes_groups_df(adata, group = None,
        pval_cutoff=0.001,log2fc_min=1)
    markersDF1.to_csv(f"rank_genes_groups_{res:4.2f}.markerGenes.csv", index=False)


adata.write(results_file + 'scRNAseq_Mm_TKOIL10_processing2.h5ad')



for res in [0.15, 0.18, 0.2, 0.25, 0.3]:
    sc.tl.leiden(adata, key_added=f"leiden_res_{res:4.2f}", resolution=res, n_iterations = 2, flavor="igraph")
    sc.pl.umap(adata, color=[f"leiden_res_{res:4.2f}"], show=False, legend_loc = 'on data',
        save=f"leiden_res_{res:4.2f}"+"_clustering.testRes.UMAP.png")

# adata.write(results_file + 'scRNAseq_Mm_TKOIL10_processing2_temp.h5ad')

# adata = sc.read_h5ad(results_file+"scRNAseq_Mm_TKOIL10_processing2_temp.h5ad")


adata.obs['leiden_res_last'] = adata.obs['leiden_res_0.18']

adata.obs['leiden_res_last'] = adata.obs['leiden_res_last'].cat.add_categories(['11', '12', '13'])
adata.obs.loc[adata.obs['leiden_res_0.30'] == '2','leiden_res_last'] = '11'
adata.obs.loc[adata.obs['leiden_res_0.30'] == '5','leiden_res_last'] = '12'
adata.obs.loc[adata.obs['leiden_res_0.30'] == '13','leiden_res_last'] = '13'

sc.pl.umap(adata, color=['leiden_res_last'], show=False, legend_loc = 'on data',
    save=f"leiden_res_last_clustering.UMAP.png")

sc.tl.rank_genes_groups(adata, groupby="leiden_res_last", 
    method="wilcoxon", use_raw = True)
sc.pl.rank_genes_groups_dotplot(adata, groupby="leiden_res_last", 
    standard_scale="var", n_genes=5,
    show=False, save=f"dotplot_markers.leiden_res_last.cluster.markerGenes.png")
markersDF1 = sc.get.rank_genes_groups_df(adata, group = None,
    pval_cutoff=0.001,log2fc_min=1)
markersDF1.to_csv(f"rank_genes_groups_last.markerGenes.csv", index=False)

adata.write(results_file + 'scRNAseq_Mm_TKOIL10_processing2_temp.h5ad')



marker_genes = {
    "test":['Il10'],
    "Tcell": ["Cd3e",'Cd3g','Trbc2','Trac','Ets1','Rora'],
    "NK": ['Ncr1','Klrb1','Klrd1','Gzma','Gzmb','Gzmc'], # 没有Cd3
    "γδT":['Trdc','Trgc1','Trgc2','Trgc3','Trgc4'],
    'ILC2':['Gata3','Rora','Il1rl1','Il2ra'], # 1
    'ILC3':['Rorc'],
    'CD4T': ["Cd4", ],
    'Macro':['Csf1r','Zeb2','Lyz2'],
    'Neu':['S100a8','S100a9','Ly6g'],
    'Goblet':['Muc2','Tff3','Fcgbp','Agr2'],
    'Mono':['Ly6c2','Ccr2','Csf2rb'],
    'Fib':[],
    'Epithelial':['Epcam','Krt19','Krt8'],
    'Stromal':['Pdgfra','Wwtr1','Tcf4','Igfbp7'],
    'Myofib':['Cald1','Csrp1','Acta2','Tagln','Myh11'],
    'B':['Cd79a','Ms4a1']
}

sc.pl.dotplot(adata, marker_genes, groupby="leiden_res_last", 
    standard_scale="var", use_raw = True, show=False, 
    save="dotplot_markers.leiden_res_last_markerGenes.png")

adata = adata[adata.obs['leiden_res_last']!='13'].copy()


adata.obs["cell_type_test"] = adata.obs["leiden_res_last"].map(
    {
        "0": "NK",
        "1": "ILC2",
        "2": "T",
        "3": "Macro/Mono",
        "4": "Neutrophil",
        "5": "Goblet",
        "6": "Un-know",
        "7": "Stromal",
        "8": "Epithelial",
        "9": "Stromal",
        "10": "Stromal",
        "11": "T",
        "12": "Epithelial"
    }
)


marker_genes1 = {
    "Tcell": ["Cd3e",'Cd3g','Trbc2','Trac','Ets1','Rora','Cd4','Cd8a'],
    'B':['Cd79a','Ms4a1'],
    "NK": ['Ncr1','Klrb1','Klrd1','Gzma','Gzmb','Gzmc'], # 没有Cd3
    'ILC2':['Gata3','Rora','Il1rl1','Il2ra'], # 1
    'Macro':['Csf1r','Zeb2','Lyz2'],
    'Neu':['S100a8','S100a9','Ly6g'],
    'Mono':['Ly6c2','Ccr2','Csf2rb'],
    'Ery':['Hbb-bt','Pf4','Ppbp'],
    'pDC':['Siglech','Nrp1'],
    'Epithelial':['Epcam','Krt19','Krt8'],
    'Goblet':['Muc2','Tff3','Fcgbp','Agr2'],
    'Stromal':['Pdgfra','Wwtr1','Tcf4','Igfbp7']
}

sc.pl.dotplot(adata, marker_genes1, groupby="cell_type_test", 
    standard_scale="var", use_raw = True, show=False, 
    save="dotplot_markers.cell_type_test_markerGenes1.png")

sc.pl.umap(adata, color=["cell_type_test"],
    size=2, show=False, wspace=0.5,hspace=0.5,ncols=3,
    save="UMAP.plot_cell_type_test.png")


adata = adata[adata.obs['leiden_res_last']!='6'].copy()

adata.obs["cell_type_test"] = adata.obs["leiden_res_last"].map(
    {
        "0": "Cd8T",
        "1": "ILC2",
        "2": "Cd4T",
        "3": "Macro",
        "4": "Neutrophil",
        "5": "Goblet",
        "7": "Stromal",
        "8": "Epithelial",
        "9": "Stromal",
        "10": "Stromal",
        "11": "Cd4T",
        "12": "Epithelial"
    }
)

marker_genes2 = {
    "T": ["Cd3e",'Cd3g','Trac','Trbc2'],
    "Cd4T": ['Cd4'],
    "Cd8T": ['Cd8a'],
    'Epithelial':['Epcam','Krt19','Krt8'],
    'Goblet':['Muc2','Tff3','Fcgbp','Agr2'],
    'ILC2':['Gata3','Il1rl1','Il2ra'],
    'Macro':['Adgre1','Cx3cr1','Cd68','Csf1r'],
    # "NK": ['Ncr1','Klrb1','Klrd1','Gzma','Gzmb','Gzmc'], # 没有Cd3
    'Neu':['S100a8','S100a9','Ly6g'],
    'Stromal':['Pdgfra','Wwtr1','Tcf4','Igfbp7'],
    'B':['Cd79a','Ms4a1'],
    'Mono':['Ly6c2','Ccr2','Csf2rb'],
    }

sc.pl.dotplot(adata, marker_genes2, groupby="cell_type_test", 
    standard_scale="var", use_raw = True, show=False, 
    save="dotplot_markers.cell_type_test_markerGenes2.png")

sc.pl.umap(adata, color=["cell_type_test"],
    size=2, show=False, wspace=0.5,hspace=0.5,ncols=3,
    save="UMAP.plot_cell_type_test.step3.png")

adata.write(results_file + 'scRNAseq_Mm_TKOIL10_processing3.h5ad')
# adata = sc.read_h5ad(results_file+"scRNAseq_Mm_TKOIL10_processing4.h5ad")


groups = adata.obs['Group'].unique().tolist()
fig, axs = plt.subplots(1, 2, figsize=(4 * 2, 4))
plt.subplots_adjust(hspace=0.4, wspace=0.4)
for idx, i in enumerate(groups):
    tempAdata = adata[adata.obs['Group'] == i].copy()
    sc.pl.umap(tempAdata, color=['Il10'], frameon = False, # legend_fontsize=7.5,
               legend_loc = 'right margin', ncols = 2, ax=axs[idx], show=False,vmax=4.5,
               title=i,
               color_map = 'Reds')
    del(tempAdata)
plt.tight_layout()
plt.savefig("251201.UMAP.plot.allCells_Il10.pdf", dpi=150, bbox_inches='tight')
plt.close()


Tcell = adata[adata.obs['cell_type_test'].isin(['Cd4T','Cd8T','Macro/Mono/DC'])].copy()
Tcell.obs['Group2'] = Tcell.obs['Group'].astype(str) + '_' + Tcell.obs['cell_type2'].astype(str)

sc.pl.dotplot(Tcell, ['Il10'], groupby="Group2", 
    standard_scale="var", use_raw = True, show=False, 
    save="dotplot_markers.Tcell_Il10.png")

sc.pl.dotplot(Tcell, ['Il10'], groupby="Group2", 
    standard_scale="var", use_raw = True, show=False, 
    save="dotplot_markers.Tcell_Il10.2.pdf")

Tcell.obs['Group3'] = Tcell.obs['cell_type2'].astype(str) + '_' + Tcell.obs['Group'].astype(str)
sc.pl.dotplot(Tcell, ['Il10'], groupby="Group3", 
    standard_scale="var", use_raw = True, show=False, 
    save="dotplot_markers.Tcell_Il10.3.pdf")

sc.pl.umap(adata, color=["cell_type_test","Cd4","Cd8a","Cd3e",'Cd3g','Ncr1','Gzma'],
    size=2, show=False, wspace=0.5, hspace=0.5, ncols=3,
    save="UMAP.plot_cell_type_test.Cd4_Cd8a.step3.png")


Macro = adata[adata.obs['cell_type_test'].isin(['Macro'])].copy()
Macro.obs['Group2'] = Macro.obs['Group'].astype(str) + '_' + Macro.obs['cell_type_test'].astype(str)

sc.pl.dotplot(Macro, ['Il10'], groupby="Group2", 
    standard_scale="var", use_raw = True, show=False, 
    save="dotplot_markers.Macrocell_Il10.png")

sc.pl.dotplot(Macro, ['Il10'], groupby="Group2", 
    standard_scale="var", use_raw = True, show=False, 
    save="dotplot_markers.Macrocell_Il10.pdf")


adata.obs['cell_type_test'] = adata.obs['cell_type_test'].astype(str)
adata.obs['cell_type_test'].unique()

adata.obs["cell_type_test"] = adata.obs["leiden_res_last"].map(
    {
        "0": "Cd8T",
        "1": "ILC2",
        "2": "Cd4T",
        "3": "Macro/Mono/DC",
        "4": "Neutrophil",
        "5": "Goblet",
        "7": "Stromal",
        "8": "Epithelial",
        "9": "Stromal",
        "10": "Stromal",
        "11": "Cd4T",
        "12": "Epithelial"
    }
)

marker_genes2 = {
    "T": ["Cd3e",'Cd3g','Trac','Trbc2'],
    "Cd4T": ['Cd4'],
    "Cd8T": ['Cd8a'],
    'Epithelial':['Epcam','Krt19','Krt8'],
    'Goblet':['Muc2','Tff3','Fcgbp','Agr2'],
    'ILC2':['Gata3','Il1rl1','Il2ra'],
    'Macro':['Adgre1','Cx3cr1','Cd68','Csf1r'],
    # "NK": ['Ncr1','Klrb1','Klrd1','Gzma','Gzmb','Gzmc'], # 没有Cd3
    'Neu':['S100a8','S100a9','Ly6g'],
    'Stromal':['Pdgfra','Wwtr1','Tcf4','Igfbp7'],
    'B':['Cd79a','Ms4a1'],
    'Mono':['Ly6c2','Ccr2','Csf2rb'],
    }

sc.pl.dotplot(adata, marker_genes2, groupby="cell_type_test", 
    standard_scale="var", use_raw = True, show=False, 
    save="dotplot_markers.cell_type_test_markerGenes2.png")

sc.pl.dotplot(adata, marker_genes2, groupby="cell_type_test", 
    standard_scale="var", use_raw = True, show=False, 
    save="dotplot_markers.cell_type_test_markerGenes2.pdf")

sc.pl.umap(adata, color=["cell_type_test"],
    size=2, show=False, wspace=0.5,hspace=0.5,ncols=3,
    save="UMAP.plot_cell_type_test.step3.png")

sc.pl.umap(adata, color=["cell_type_test"],
    size=2, show=False, wspace=0.5,hspace=0.5,ncols=3,
    save="UMAP.plot_cell_type_test.step3.pdf")

adata.write(results_file + 'scRNAseq_Mm_TKOIL10_processing4.h5ad')

Macro = adata[adata.obs['cell_type_test'].isin(['Macro/Mono/DC'])].copy()
Macro.obs['Group2'] = Macro.obs['Group'].astype(str) + '_' + Macro.obs['cell_type_test'].astype(str)

sc.pp.neighbors(Macro, use_rep='X_scVI', n_neighbors = 30, knn=True)
sc.tl.umap(Macro, min_dist = 0.5)
sc.tl.leiden(Macro, key_added=f"Macro_leiden_res_0.22", resolution=0.22, 
    n_iterations = 2, flavor="igraph")
sc.pl.umap(Macro, color=[f"Macro_leiden_res_0.22"], show=False, legend_loc = 'on data',
    save=f"251201.UMAP_Macro_leiden_res_0.22_clustering.step4.png")
sc.tl.rank_genes_groups(Macro, groupby=f"Macro_leiden_res_0.22", method="wilcoxon", use_raw = True)
sc.pl.rank_genes_groups_dotplot(Macro, groupby=f"Macro_leiden_res_0.22", standard_scale="var", n_genes=5,
    show=False, save=f"2501201.dotplot_markers.Macro_leiden_res_0.22.step4.png")
markersDF1 = sc.get.rank_genes_groups_df(Macro, group = None, pval_cutoff=0.001, log2fc_min=1)
markersDF1.to_csv(f"251201.rank_genes_groups_Macro_leiden_res_0.22.step4.csv", index=False)


marker_genes = {
    'test':['Il10','Tnf'],
    "Macro_TR": ["Zeb2",'Abca1','Lrmda','Trps1','mt-Rnr1', 'mt-Rnr2'],
    "M1": ['Cd14','Il1b','Clec7a'],
    "M2":['Mrc1','C1qb','Ccl6','Ccl9'],
    "Mono":['Ly6c2'],
    "Macro":['Adgre1','Cx3cr1','Cd68','Csf1r'],
    "DC":['Xcr1','Clec9a','Itgae','H2-Eb1','H2-Ab1','H2-Aa'],
    "Mix-del":['Cd3e','Cd3g','Ets1','Trbc2','Trac','Cd4','Cd8a']
}

sc.pl.dotplot(Macro, marker_genes, groupby="Macro_leiden_res_0.22", 
    standard_scale="var", use_raw = True, show=False, 
    save="2501201.dotplot_markers.Macro_leiden_res_0.22_markerGenes.png")


Macro.obs["Macro_sub"] = Macro.obs["Macro_leiden_res_0.22"].map(
    {
        "0": "?",
        "1": "Mono",
        "2": "Macro",
        "3": "Mix-del",
        "4": "cDC1",
    }
)
 
sc.pl.dotplot(Macro, marker_genes, groupby="Macro_sub", 
    standard_scale="var", use_raw = True, show=False, 
    save="2501201.dotplot_markers.Macro_sub_markerGenes.step4.png")
sc.pl.umap(Macro, color=['Il10','Tnf','Il1b','Adgre1','Xcr1','Ly6c2',"Macro_sub"], 
    show=False, legend_loc = 'right margin',
    save=f"251201.UMAP_Macro_sub_clustering.step4.png")

Macro = Macro[~Macro.obs['Macro_sub'].isin(['Mix-del'])]

sc.pp.neighbors(Macro, use_rep='X_scVI', n_neighbors = 10, knn=True)
sc.tl.umap(Macro, min_dist = 0.5)
sc.tl.leiden(Macro, key_added=f"Macro_leiden_res_0.30", resolution=0.30, 
    n_iterations = 2, flavor="igraph")
sc.pl.umap(Macro, color=[f"Macro_leiden_res_0.30"], show=False, legend_loc = 'on data',
    save=f"251201.UMAP_Macro_leiden_res_0.30_clustering.step4.rmDel.png")
sc.tl.rank_genes_groups(Macro, groupby=f"Macro_leiden_res_0.30", method="wilcoxon", use_raw = True)
sc.pl.rank_genes_groups_dotplot(Macro, groupby=f"Macro_leiden_res_0.30", standard_scale="var", n_genes=5,
    show=False, save=f"2501201.dotplot_markers.Macro_leiden_res_0.30.step4.rmDel.png")
markersDF1 = sc.get.rank_genes_groups_df(Macro, group = None, pval_cutoff=0.001, log2fc_min=1)
markersDF1.to_csv(f"251201.rank_genes_groups_Macro_leiden_res_0.30.step4.rmDel.csv", index=False)

marker_genes = {
    'test':['Il10','Mertk','Cd209a'],
    "Macro_TR": ["Zeb2",'Abca1','Lrmda','Trps1','mt-Rnr1', 'mt-Rnr2'],
    "M1": ['Cd14','Il1b','Clec7a'],
    "M2":['Mrc1','C1qb','Ccl6','Ccl9'],
    "Mono":['Ly6c2','Itgam','Sell','Hp','Ccr2','Cx3cr1','Fcgr4','Pparg'],
    "Macro":['Adgre1','Cx3cr1','Cd68','Csf1r'],
    "cDC":['Xcr1','Clec9a','Itgae','Irf4','Ccr7','H2-Eb1','H2-Ab1','H2-Aa'],
    "Mix-del":['Cd3e','Cd3g','Ets1','Trbc2','Trac','Cd4','Cd8a']
    }

sc.pl.dotplot(Macro, marker_genes, groupby="Macro_leiden_res_0.30", 
    standard_scale="var", use_raw = True, show=False, 
    save="2501201.dotplot_markers.Macro_leiden_res_0.30_markerGenes.step4.rmDel.png")

sc.pl.umap(Macro, color=['Tnf','Mertk','Il10','Irf4','Xcr1','Cxcr1','Adgre1','Ly6c2',f"Macro_leiden_res_0.30"], 
    show=False, legend_loc = 'right margin',
    save=f"251201.UMAP_Macro_leiden_res_0.30_clustering.step4.rmDel.png")


Macro.obs["Macro_sub"] = Macro.obs["Macro_leiden_res_0.30"].map(
    {
        "0": "cDC",
        "1": "Mono-Macro",
        "2": "Cx3cr1+ Macro",
        "3": "Tnf hi Mono",
        "4": "Mertk hi Macro",
        "5": "Ly6c2 hi Mono",
    }
)

orderCell=["cDC","Mono-Macro","Cx3cr1+ Macro","Mertk hi Macro","Tnf hi Mono","Ly6c2 hi Mono"]
Macro.obs['Macro_sub'] = pd.Categorical(Macro.obs['Macro_sub'], categories=orderCell, ordered=True)

marker_genes = {
    "cDC":['Xcr1','Clec9a','Itgae','Irf4','Ccr7'],
    "Mono-Macro": ["Zeb2",'Trps1','Abca1','Lrmda','mt-Rnr1', 'mt-Rnr2'],
    'Cx3cr1+ Macro':['Cx3cr1'],
    "Mertk hi Macro": ['Adgre1','Mertk','Mrc1','C1qb','Ccl6','Ccl9','Il10'],
    'Tnf+ Mono':['Ly6c2','Tnf'],
    "Ly6c2 hi Mono":['Cd68','Fcgr4'],
}

sc.pl.dotplot(Macro, marker_genes, groupby="Macro_sub", 
    standard_scale="var", use_raw = True, show=False, 
    save="2501201.dotplot_markers.Macro_sub_markerGenes.step4.rmDel.png")

sc.pl.dotplot(Macro, marker_genes, groupby="Macro_sub", 
    standard_scale="var", use_raw = True, show=False, 
    save="2501201.dotplot_markers.Macro_sub_markerGenes.step4.rmDel.pdf")

sc.pl.umap(Macro, color=['Tnf','Mertk','Il10','Adgre1','Ly6c2',f"Macro_sub"], show=False, legend_loc = 'right margin',
    save=f"251201.UMAP_Macro_sub_clustering.step4.rmDel.png")


groups = Macro.obs['Group'].unique().tolist()
fig, axs = plt.subplots(1, 2, figsize=(6 * 2, 4))
plt.subplots_adjust(hspace=0.4, wspace=0.4)
for idx, i in enumerate(groups):
    tempAdata = Macro[Macro.obs['Group'] == i].copy()
    sc.pl.umap(tempAdata, color=['Macro_sub'], frameon = False, # legend_fontsize=7.5,
               legend_loc = 'right margin', ncols = 2, ax=axs[idx], show=False, # vmax=4.5,
               title=i) #, color_map = 'Reds')
    del(tempAdata)
plt.tight_layout()
plt.savefig("251201.UMAP.plot.allCells_celltypeMacro.png", dpi=150, bbox_inches='tight')
plt.close()

# Macro.write(results_file+f'251201.Macro_subcluster.step4.h5ad')

## merge the last processing 4 h5ad
adata.obs['cell_type2'] = adata.obs['cell_type_test'].astype(str)
tt = adata.obs
tt.loc[Macro.obs.index,'cell_type2'] = Macro.obs.loc[Macro.obs.index,'Macro_sub'].astype(str)
adata.obs = tt
adata = adata[~adata.obs['cell_type2'].isin(['Macro/Mono/DC'])].copy()

orderCell=['Cd4T', 'Cd8T', 'ILC2',  'cDC', 'Mono-Macro', 'Cx3cr1+ Macro', 'Mertk hi Macro', 'Tnf hi Mono', 'Ly6c2 hi Mono',
'Neutrophil', 'Stromal', 'Epithelial', 'Goblet']

adata.obs['cell_type2'] = pd.Categorical(adata.obs['cell_type2'], categories=orderCell, ordered=True)
adata.write(results_file + 'scRNAseq_Mm_TKOIL10_processing4.h5ad')


## cell counts
count_df = adata.obs.groupby(['cell_type2', 'Group'],observed=False).size().reset_index(name='cell_count')
total_cells = count_df['cell_count'].sum()
count_df['proportion'] = count_df['cell_count'] / total_cells

count_df = pd.crosstab(
    adata.obs['cell_type2'], 
    adata.obs['Group'],
    margins=True,
    margins_name='Total'
)

proportion_df = pd.crosstab(
    adata.obs['cell_type2'], 
    adata.obs['Group'],
    normalize='columns'
)

# plot the cell fraction in R
tempData = adata[adata.obs['sample']=='LC_1',].copy()
pd.crosstab(
    adata.obs['cell_type2'], 
    adata.obs['sample'],
    normalize='columns'
)



## analysis human_efferocytosis
Macro = sc.read_h5ad(results_file+"251201.Macro_subcluster.step4.h5ad")
sc.pl.umap(Macro, color=[f"Macro_sub"], show=False, legend_loc = 'right margin',
    save=f"251201.UMAP_Macro_sub_clustering.step4.rmDel.pdf")

human_efferocytosis_genes = { 
    "Core_Receptors": [ "Mertk", "Axl", "Tyro3", "Timd4", "Cd36", "Msr1" ], 
    "Bridging_Ligands": [ "Gas6", "Pros1" ], 
    "Phagocytosis_Machinery": [ "Rac1", "Cdc42", "Rhoa", "Dock1", "Vav1", "Crkl" ], 
    "Pro_Resolution_M2": [ "Il10", "Tgfb1", "Arg1", "Mrc1", "Retnla", "Alox15", "Sirpa"], 
    "Metabolism_Lipid_Efflux": [ "Abca1", "Ldlr", "Pparg", "Ppargc1a", "Acat1" ], 
    "Digestion_Lysosomal": [ "Ctss", "Ctsb", "Lamp1", "Atp6v0d1"], 
    "Apoptotic_Cell_Signals": [ "Cd47", "Cxcl10"] }

Macro.obs['Group2'] = Macro.obs['Group'].astype(str) + '_' + Macro.obs['Macro_sub'].astype(str)
orderCell=['WT_cDC', 'TKO_cDC',
'WT_Mono-Macro', 'TKO_Mono-Macro',
'WT_Cx3cr1+ Macro', 'TKO_Cx3cr1+ Macro',
'WT_Mertk hi Macro', 'TKO_Mertk hi Macro',
'WT_Tnf hi Mono', 'TKO_Tnf hi Mono',
'WT_Ly6c2 hi Mono', 'TKO_Ly6c2 hi Mono']

Macro.obs['Group2'] = pd.Categorical(Macro.obs['Group2'], categories=orderCell, ordered=True)

sc.pl.dotplot(Macro, 
    human_efferocytosis_genes, groupby="Group2", 
    standard_scale="var", use_raw = True, show=False, 
    save="2501201.dotplot_markers.human_efferocytosis_genes_markerGenes.step4.rmDel.png")

human_efferocytosis_genes2 = {
    "Phagocytosis":['Timd4', 'Cd36', 'Gas6', 'Pros1', 'Dock1', 'Crkl', 'Tgfb1', 'Retnla',
    'Mfge8', 'Itgav', 'Itgb3', 'Trem2', 'Rac1','Rac2', 'Elmo1', 'Elmo2', 'Akt1', 'Pik3ca', 'Socs1', 'Socs3']
}

sc.pl.dotplot(Macro, 
    human_efferocytosis_genes2, groupby="Group2", 
    standard_scale="var", use_raw = True, show=False, 
    save="2501201.dotplot_markers.human_efferocytosis_genes_markerGenes2.step4.rmDel.png")


## differential genes
for i in Macro.obs['Macro_sub'].unique().tolist():
    tempAdata = Macro[Macro.obs['Macro_sub'].isin([str(i)]),:].copy()
    sc.tl.rank_genes_groups(tempAdata, groupby="Group", groups= ['TKO'], reference = 'WT', method="wilcoxon", use_raw = True)
    markersDF1 = sc.get.rank_genes_groups_df(tempAdata, group = None)
    markersDF1.to_csv(f"2501201.rank_genes_groups_TKOvsWT_{i}.csv", index=False)


folder_path = '/public/home/chenjiaminggroup/wufan/20251111_scRNAseq_MouseModel_YJW/DEGfiles/'
dataframes = []
aa = Macro.obs['Macro_sub'].unique().tolist()
for i in aa:
    filename = f"2501201.rank_genes_groups_TKOvsWT_{i}.csv"
    file_path = os.path.join(folder_path, filename)
    df = pd.read_csv(file_path)
    df['Type'] = 'TKOvsWT'
    df['cellName'] = i
    dataframes.append(df)

combined_df = pd.concat(dataframes, ignore_index=True)
output_file = 'MergeAll.DEG.tissue.csv'

# pval_cutoff=0.05
# combined_df['DEG'] = 'NO'
# combined_df.loc[(combined_df['logfoldchanges'] >= 1) & (combined_df['pvals'] < 0.05),'DEG'] = 'UP'
# combined_df.loc[(combined_df['logfoldchanges'] <= -1) & (combined_df['pvals'] < 0.05),'DEG'] = 'DOWN'
import numpy as np
combined_df['DEG'] = 'NO'
combined_df.loc[(combined_df['logfoldchanges'] > np.log2(1.5)) & (combined_df['pvals'] < 0.05),'DEG'] = 'UP'
combined_df.loc[(combined_df['logfoldchanges'] < -1*np.log2(1.5)) & (combined_df['pvals'] < 0.05),'DEG'] = 'DOWN'

# pd.set_option('display.max_rows', 100) 
coumtMatrix = combined_df.groupby(['Type', 'cellName','DEG']).size().reset_index(name='count')
coumtMatrix.to_csv(os.path.join(folder_path, 'MergeAll.DEG.tissue.countMatrix.csv'), index=False)
combined_df.to_csv(os.path.join(folder_path, output_file), index=False)

df = combined_df[combined_df['DEG'] != 'NO']
df.groupby(['Type', 'cellName']).size().reset_index(name='count')
tt = df.pivot_table(index='cellName', columns='Type', aggfunc='size', fill_value=0)
tt.to_csv(os.path.join(folder_path, 'MergeAll.DEG.tissue.countMatrix2.csv'), index=False)
del tt

df.to_csv(os.path.join(folder_path, "MergeAll.DEG.tissue.2.csv"), index=False)


## GESA
from gseapy import Msigdb
import gseapy as gp
import json

# msig = Msigdb()
# gmt = msig.get_gmt(category='m5.go.bp',dbver='2025.1.Mm')
# with open('/public/home/chenjiaminggroup/wufan/software/MSigDB/m5.go.bp.gmt.json', 'w', encoding='utf-8') as f:
#     json.dump(gmt, f, ensure_ascii=False, indent=4)  # indent 用于美化格式

with open('/public/home/chenjiaminggroup/wufan/software/MSigDB/m5.go.bp.gmt.json', 'r', encoding='utf-8') as f:
    gmt = json.load(f)

print(gmt['GOBP_ENGULFMENT_OF_APOPTOTIC_CELL'])
print(gmt['GOBP_PHAGOCYTOSIS_ENGULFMENT'])

for key in gmt.keys():
    print(key)


# bdata = Macro[Macro.obs['Macro_sub']=='Mertk hi Macro']
raw_adata = bdata.raw.to_adata()  # 转换为完整的 AnnData 对象
res = gp.gsea(data=raw_adata.to_df().T, # row -> genes, column-> samples
        gene_sets=gene_sets,
        cls=bdata.obs.Group,
        permutation_num=1000,
        permutation_type='phenotype',
        outdir=None,
        method='s2n', # signal_to_noise
        threads= 16)

gene_sets = {
    "GOBP_PHAGOCYTOSIS_ENGULFMENT":gmt['GOBP_PHAGOCYTOSIS_ENGULFMENT'],
    "GOBP_CANONICAL_INFLAMMASOME_COMPLEX_ASSEMBLY":gmt['GOBP_CANONICAL_INFLAMMASOME_COMPLEX_ASSEMBLY'],
    "GOBP_APOPTOTIC_CELL_CLEARANCE":gmt['GOBP_APOPTOTIC_CELL_CLEARANCE']
}

degs = combined_df.loc[combined_df['cellName']=='Mertk hi Macro',]
pre_res = gp.prerank(degs.loc[:,['names', 'logfoldchanges']], gene_sets=gmt)
pre_res_df = pre_res.res2d
pre_res_df = pre_res_df.sort_values(by='NOM p-val')
pre_res_df['Term'][20:50].to_list()


## Moudle score

InflammatoryResponse = [
    "CXCL6","CSF3","CXCL8","CD82","ATP2A2","ADM","PVR","ICAM1","BEST1","RGS1","NAMPT","PDE4B","BDKRB1","TIMP1","PTGIR",
    "IL4R","RIPK2","SPHK1","RHOG","PLAUR","OSMR","F3","EREG","MMP14","SELENOS","IL1B","IRF7","LCP2","ITGA5","RAF1",
    "CHST2","PTGER2","HIF1A","HRH1","CCL7","PDPN","LDLR","CD55","ABCA1","GCH1","TNFSF15","CCL20","SLC31A1","PCDH7",
    "ADRM1","NMI","INHBA","ATP2B1","DCBLD2","BST2","NFKBIA","CXCL10","PTPRE","IL6","KLF6","IL7R"]


IR_mouse = [
"Cxcl6", "Csf3", "Cxcl8", "Cd82", "Atp2a2",
"Adm", "Pvr", "Icam1", "Best1", "Rgs1",
"Nampt", "Pde4b", "Bdkrb1", "Timp1", "Ptgir",
"Il4r", "Ripk2", "Sphk1", "Rhog", "Plaur",
"Osmr", "F3", "Ereg", "Mmp14", "Selenos",
"Il1b", "Irf7", "Lcp2", "Itga5", "Raf1",
"Chst2", "Ptger2", "Hif1a", "Hrh1", "Ccl7",
"Pdpn", "Ldlr", "Cd55", "Abca1", "Gch1",
"Tnfsf15", "Ccl20", "Slc31a1", "Pcdh7",
"Adrm1", "Nmi", "Inhba", "Atp2b1", "Dcbld2",
"Bst2", "Nfkbia", "Cxcl10", "Ptpre", "Il6",
"Klf6", "Il7r"]


efferocytosis = [
    "CD24","NR1H3","ADAM10","SIRPB1","ATP8A1","MERTK","CEBPB","XKR4","ANO4","CPT1C","CPT1A","CPT1B","CREB1","CRK","CRKL",
    "SIRPA","MAPK14","CX3CR1","TMEM30B","AGER","DNMT3A","DOCK1","DUSP2","DUSP4","DUSP5","DUSP7","DUSP8","ABCA1","S1PR1",
    "ANO6","ANO5","EPO","EPOR","STAB1","ATP11B","ATP11A","VPS39","VPS8","SIRT1","PLA2G15","ALOX5","PANX1","ALOX15",
    "GAS6","HAVCR1","VPS41","XKR6","ATP11C","GPR132","HIF1A","RAB7B","ANO9","XKR7","IL10","ITGAV","ITGB3","ITGB5","JAK2",
    "ARG1","ARG2","XKR9","LIPA","LRP1","ARNT","MFGE8","NFATC1","NFATC2","NFATC3","NFATC4","ATP2A1","ATP2A2","ATP2A3",
    "ODC1","P2RY2","P2RY6","ANO7","PBX1","GULP1","RAB14","PECAM1","ATP8A2","PPARD","PPARG","SLC66A1","XKR8","SIRPG",
    "STAB2","TMEM30A","AXL","VPS11","MAPK1","MAPK3","MAPK11","MAPK13","MAP2K1","MAP2K2","PROS1","SPHK2","PTGER2","PTGER4",
    "PTGS2","PTK2","ADGRB1","VPS18","PTPN6","PTPN11","RAB5A","RAB5B","RAB5C","RAC1","RXRA","MAPK12","CX3CL1","ANO3",
    "RAB17","SGK1","VPS16","P2RY12","VPS33A","SLC2A1","SLC16A1","BSG","ADAM17","TGFB1","THBS1","C1QA","C1QB","C1QC",
    "TYRO3","NR1H2","UQCRFS1","RAB7A","DUSP16","CALR","CAMK2A","CAMK2B","CAMK2D","CAMK2G","CASP1","CASP3","PLA2G6",
    "CASP7","MEGF11","MEGF10","HAVCR2","SCARF1","ADAM9","SPHK1","SIGLEC10","CH25H","TIMD4","MAPKAPK2","TGFBRAP1","CD36",
    "BCAR1","CD47","ELMO1"]

efferocytosis_mouse = [
    "Cd24", "Nr1h3", "Adam10", "Sirpb1", "Atp8a1", "Mertk",
    "Cebpb", "Xkr4", "Ano4", "Cpt1c", "Cpt1a", "Cpt1b",
    "Creb1", "Crk", "Crkl", "Sirpa", "Mapk14", "Cx3cr1",
    "Tmem30b", "Ager", "Dnmt3a", "Dock1", "Dusp2", "Dusp4",
    "Dusp5", "Dusp7", "Dusp8", "Abca1", "S1pr1", "Ano6",
    "Ano5", "Epo", "Epor", "Stab1", "Atp11b", "Atp11a",
    "Vps39", "Vps8", "Sirt1", "Pla2g15", "Alox5", "Panx1",
    "Alox15", "Gas6", "Havcr1", "Vps41", "Xkr6", "Atp11c",
    "Gpr132", "Hif1a", "Rab7b", "Ano9", "Xkr7", "Il10",
    "Itgav", "Itgb3", "Itgb5", "Jak2", "Arg1", "Arg2",
    "Xkr9", "Lipa", "Lrp1", "Arnt", "Mfge8", "Nfatc1",
    "Nfatc2", "Nfatc3", "Nfatc4", "Atp2a1", "Atp2a2", "Atp2a3",
    "Odc1", "P2ry2", "P2ry6", "Ano7", "Pbx1", "Gulp1",
    "Rab14", "Pecam1", "Atp8a2", "Ppard", "Pparg", "Slc66a1",
    "Xkr8", "Sirpg", "Stab2", "Tmem30a", "Axl", "Vps11",
    "Mapk1", "Mapk3", "Mapk11", "Mapk13", "Map2k1", "Map2k2",
    "Pros1", "Sphk2", "Ptger2", "Ptger4", "Ptgs2", "Ptk2",
    "Adgrb1", "Vps18", "Ptpn6", "Ptpn11", "Rab5a", "Rab5b",
    "Rab5c", "Rac1", "Rxra", "Mapk12", "Cx3cl1", "Ano3",
    "Rab17", "Sgk1", "Vps16", "P2ry12", "Vps33a", "Slc2a1",
    "Slc16a1", "Bsg", "Adam17", "Tgfb1", "Thbs1", "C1qa",
    "C1qb", "C1qc", "Tyro3", "Nr1h2", "Uqcrfs1", "Rab7a",
    "Dusp16", "Calr", "Camk2a", "Camk2b", "Camk2d", "Camk2g",
    "Casp1", "Casp3", "Pla2g6", "Casp7", "Megf11", "Megf10",
    "Havcr2", "Scarf1", "Adam9", "Sphk1", "Siglec10", "Ch25h",
    "Timd4", "Mapkapk2", "Tgfbrap1", "Cd36", "Bcar1", "Cd47",
    "Elmo1"]

efferocytosis_mouse2 = ['Timd4', 'Cd36', 'Rubcn', 'C1qtnf1', 'Lyve1', 'Retnla']

gene_set_names = ['Inflammatory_Response', 'Efferocytosis']

sc.tl.score_genes(
    adata, 
    gene_list=efferocytosis_mouse,
    ctrl_size=50,
    n_bins=25,
    score_name="efferocytosis_mouse",
    random_state=210
)

sc.tl.score_genes(
    adata, 
    gene_list=efferocytosis_mouse2,
    ctrl_size=50,
    n_bins=25,
    score_name="efferocytosis_mouse2",
    random_state=210
)

sc.tl.score_genes(
    adata, 
    gene_list=IR_mouse,
    ctrl_size=50,
    n_bins=25,
    score_name="IR_mouse",
    random_state=210
)

module_score_df = adata.obs[['sample','Group','cell_type_test', 'cell_type2', 'IR_mouse', 'efferocytosis_mouse','efferocytosis_mouse2']]
module_score_df.to_csv("./module_score.csv")


## plot heatmap;
# import marsilea as ma
# import marsilea.plotter as mp

markers={
    "Find me & Eat me": ['Cx3cr1', 'Mertk', 'Axl', 'Tyro3', 'Timd4', 'Cd36','Cd209f','Itgav', 'Itgb3', 'Stab2', 'Lrp1'],
    "Bridging Molecules": ['Gas6', 'Pros1', 'Mfge8', 'C1qa', 'C1qc', 'C1qtnf1', 'Fcna'],
    "Engulfment": ['Rac1', 'Elmo1', 'Gulp1',  'Dock1',  'Rhoa', 'Rubcn','Dnm1','Tiam1', 'Arhgap6', 'Arhgef10', 'Stard8', 'Add3', 'Tmod1'],
    'Phagolysosome Maturation': ['Atg5', 'Atg7', 'Rubcn', 'Tfeb', 'Lamp2', 'Lpl', 'Tpcn1','Ctsb', 'Ctsd','Abca1', 'Abcg1'], 
    'Resolution/M2': ['Tgfb1','Pparg','Arg1', 'Retnla', 'Igf1', 'Lyve1', 'Icosl', 'Ecm1', 'S100a4'],
    'Inflammation': ['Il1b','Tnf','Csf3','Cxcl10','Ccl3', 'Ccl4', 'Ccl5', 'Nfkbia', 'Nlrp3','Nos2','Arg2'],
}

for i in Macro.obs['Macro_sub'].unique().tolist():
    tempAdata = Macro[Macro.obs['Macro_sub'].isin([str(i)]),:].copy()
    sc.tl.rank_genes_groups(tempAdata, groupby="Group", groups= ['TKO'], reference = 'WT', method="wilcoxon", use_raw = True)
    sc.pl.rank_genes_groups_matrixplot(
        tempAdata,
        var_names=markers,
        values_to_plot="logfoldchanges",
        cmap='bwr',
        vmin=-2,
        vmax=2,
        colorbar_title='log fold change', show = False, save=f'Matrixplot_{i}_Markers.pdf')


sc.pl.matrixplot(Macro, markers, groupby='Macro_sub', 
    swap_axes=True, standard_scale = 'var',  log=False, 
    show=False, title = f'Macro subcluster',
    save=f"260201.matrixplot_DEGs_Macro.png")

sc.pl.matrixplot(Macro, markers, groupby='Macro_sub', 
    swap_axes=True, standard_scale = 'var',  log=False, 
    show=False, title = f'Macro subcluster',
    save=f"260201.matrixplot_DEGs_Macro.pdf")



