
rm(list=ls())
setwd("~/Documents/work/20240816_LiangZhu/20251111_scRNAseq_MouseModel_YJW/")

## ------- 1. DE counts -------

library(readxl)
DECounts <- read_excel("./DifferentialGeneCounts.xlsx")
DECounts <- as.data.frame(DECounts)
DECounts_plot <- DECounts[DECounts$DEG != 'NO',]
DECounts_plot$Count <- DECounts_plot$count
DECounts_plot[DECounts_plot$DEG == 'DOWN','count']  <- -1 * DECounts_plot[DECounts_plot$DEG == 'DOWN','count'] 

DECounts_plot$cellName <- factor(DECounts_plot$cellName,
                                 levels = c("Mertk hi Macro","Tnf hi Mono","Cx3cr1+ Macro",
                                            "Mono-Macro","Ly6c2 hi Mono","cDC"))

cellColLabels <- c("#F49493",
                   "#98CD87",
                   "#F47D22",
                   "#D176AE",
                   "#005AA5",
                   "#8C564B")

pdf("./plot_plus_DECounts.pdf",6,6)
ggplot(DECounts_plot, aes(cellName, count, fill=cellName))+
  geom_col(color="black", width = 0.8, linewidth = 0.8)+
  theme_bw()+
  theme(panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        panel.border = element_rect(linewidth = 1.5, colour = "black"),
        legend.title = element_blank(),
        axis.title.y.left = element_text(color="black",size=17),
        axis.text = element_text(color="black",size=15),
        axis.line = element_line(color='black'),
        axis.text.x = element_text(angle = 30, hjust = 1, vjust = 1),
        legend.position = 'none',
        plot.title = element_text(hjust = 0.5, size=20, face = 'bold'))+
  # coord_flip()+
  geom_segment(aes(x=0, y=0, xend=6.5, yend=0), linewidth = 0.7)+
  # geom_text(data = merge_res_plot[which(merge_res_plot$X_axis>0),],aes(x=Description, y=-0.1, label=Description),
  #           hjust=1, size=5)+
  # geom_text(data = merge_res_plot[which(merge_res_plot$X_axis<0),],aes(x=Description, y=0.1, label=Description),
  #           hjust=0, size=5)+
  geom_text(data = DECounts_plot[which(DECounts_plot$count>0),],aes(label=Count),
            hjust=0.5, vjust = -0.2, size=5, color='black')+
  geom_text(data = DECounts_plot[which(DECounts_plot$count<0),],aes(label=Count),
            hjust=0.5, vjust = 1.2, size=5, color="black")+
  scale_fill_manual(values = cellColLabels)+
  scale_x_discrete(expand = expansion(mult = c(0,0)))+
  ylim(-205, 70)+
  labs(x='', y='Counts', title = "Differential expression genes")
dev.off()



pdf("./plot_plus_DECounts.2.pdf",6,6)
# DECounts_plot[DECounts_plot$DEG=='UP',]
ggplot(DECounts_plot[DECounts_plot$DEG=='UP',], aes(cellName, count, fill=cellName))+
  geom_col(color="black", width = 0.8, linewidth = 0.8)+
  theme_bw()+
  theme(panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        panel.border = element_rect(linewidth = 1.5, colour = "black"),
        legend.title = element_blank(),
        axis.title.y.left = element_text(color="black",size=17),
        axis.text = element_text(color="black",size=15),
        axis.line = element_line(color='black'),
        axis.text.x = element_text(angle = 30, hjust = 1, vjust = 1),
        legend.position = 'none',
        plot.title = element_text(hjust = 0.5, size=20, face = 'bold'))+
  geom_text(data = DECounts_plot[which(DECounts_plot$count>0),],aes(label=Count),
            hjust=0.5, vjust = -0.2, size=5, color='black')+
  # geom_text(data = DECounts_plot[which(DECounts_plot$count<0),],aes(label=Count),
  #           hjust=0.5, vjust = 1.2, size=5, color="black")+
  scale_fill_manual(values = cellColLabels)+
  # scale_x_discrete(expand = expansion(mult = c(0,0)))+
  ylim(0, 70)+
  labs(x='', y='Counts', title = "Differential expression genes\n(Il10-/- / WT)")
dev.off()




## 2. volcano plot -----------------
library(readxl)
df_plot <- read.table("./MergeAll.DEG.tissue.2.csv",sep=',')
df_plot_Mertk <- df_plot[df_plot$cellName == "Mertk hi Macro",]
write.table(df_plot_Mertk, "./MergeAll.DEG.tissue.MertkhiMacro.csv", sep=',')

pdf("./plot_plus_Volcano_Mertk.pdf",7,6)
ggplot(df_plot_Mertk,aes(logfoldchanges, -log10(pvals), col = DEG)) +
  geom_point(size = 3) +
  labs(x = 'log2(TKO/WT)', y = '-log10(p_value)') +
  geom_hline(yintercept = -log10(0.05), linetype = 'dashed', linewidth = 0.5) + #横向虚线
  geom_vline(xintercept = log2(1.5), linetype = 'dashed', linewidth = 0.5) +
  geom_vline(xintercept = -log2(1.5), linetype = 'dashed', linewidth = 0.5) +
  scale_color_manual(values = c('darkblue','darkred')) + # 'grey60',
  theme_bw() +
  theme(# legend.position = 'none',
    legend.text = element_text(color = 'black',size = 12),#  family = 'Arial', face = 'plain'),
    panel.background = element_blank(),
    panel.grid = element_blank(),
    axis.text = element_text(color = 'black',size = 15),# family = 'Arial', face = 'plain'),
    # axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0),
    axis.title = element_text(color = 'black',size = 15),# family = 'Arial', face = 'plain'),
    axis.ticks = element_line(color = 'black'),
    # axis.ticks.x = element_blank(),
    axis.ticks.length = unit(.25, "cm")) +
  ggrepel::geom_text_repel(
    data = df_plot_Mertk[df_plot_Mertk$names %in% c('Il1b', 'Tnf', 'Nlrp3', 'Cebpb', 'Nos2', 'Cxcl10'),], # subset(df_final2, abs(log2FC) > 3 | -log10(p_value)>1.3),
    aes(label = names),
    size = 5, # colour = "grey50",
    box.padding = unit(1, "lines"),
    point.padding = unit(1, "lines")
  ) +
  ggrepel::geom_text_repel(
    data = df_plot_Mertk[df_plot_Mertk$names %in% c('Timd4', 'Cd36', 'Rubcn', 'C1qtnf1', 'Lyve1', 'Retnla'),], # subset(df_final2, abs(log2FC) > 3 | -log10(p_value)>1.3),
    aes(label = names),
    size = 5, # colour = "grey50",
    box.padding = unit(1, "lines")
    # point.padding = unit(1, "lines")
  )
dev.off()



df_plot_TnfMono <- df_plot[df_plot$cellName == "Tnf hi Mono",]

pdf("./plot_plus_Volcano_Tnf.pdf",7,6)
ggplot(df_plot_TnfMono,aes(logfoldchanges, -log10(pvals), col = DEG)) +
  geom_point(size = 3) +
  labs(x = 'log2(TKO/WT)', y = '-log10(p_value)') +
  geom_hline(yintercept = -log10(0.05), linetype = 'dashed', linewidth = 0.5) + #横向虚线
  geom_vline(xintercept = log2(1.5), linetype = 'dashed', linewidth = 0.5) +
  geom_vline(xintercept = -log2(1.5), linetype = 'dashed', linewidth = 0.5) +
  scale_color_manual(values = c('darkblue','darkred')) + # 'grey60',
  theme_bw() +
  theme(# legend.position = 'none',
    legend.text = element_text(color = 'black',size = 12),#  family = 'Arial', face = 'plain'),
    panel.background = element_blank(),
    panel.grid = element_blank(),
    axis.text = element_text(color = 'black',size = 15),# family = 'Arial', face = 'plain'),
    # axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0),
    axis.title = element_text(color = 'black',size = 15),# family = 'Arial', face = 'plain'),
    axis.ticks = element_line(color = 'black'),
    # axis.ticks.x = element_blank(),
    axis.ticks.length = unit(.25, "cm"))  +
  ggrepel::geom_text_repel(
    data = df_plot_TnfMono[df_plot_TnfMono$names %in% c('Lyz2', 'Ccr2'),], # subset(df_final2, abs(log2FC) > 3 | -log10(p_value)>1.3),
    aes(label = names),
    size = 5, # colour = "grey50",
    box.padding = unit(1, "lines"),
    point.padding = unit(1, "lines")
  ) # +
  # ggrepel::geom_text_repel(
  #   data = df_plot_TnfMono[df_plot_TnfMono$names %in% c('Timd4', 'Cd36', 'Rubcn', 'C1qtnf1', 'Lyve1', 'Retnla'),], # subset(df_final2, abs(log2FC) > 3 | -log10(p_value)>1.3),
  #   aes(label = names),
  #   size = 5, # colour = "grey50",
  #   box.padding = unit(1, "lines")
  #   # point.padding = unit(1, "lines"))
dev.off()


### 2.1 overlap between df_plot and some genes -----------
markers=c('Cx3cr1', 'Mertk', 'Axl', 'Tyro3', 'Timd4', 'Cd36','Cd209f','Itgav', 'Itgb3', 'Stab2', 'Lrp1',
          'Gas6', 'Pros1', 'Mfge8', 'C1qa', 'C1qc', 'C1qtnf1', 'Fcna',
          'Rac1', 'Elmo1', 'Gulp1',  'Dock1',  'Rhoa', 'Rubcn','Dnm1','Tiam1', 'Arhgap6', 'Arhgef10', 'Stard8', 'Add3', 'Tmod1',
          'Atg5', 'Atg7', 'Rubcn', 'Tfeb', 'Lamp2', 'Lpl', 'Tpcn1','Ctsb', 'Ctsd','Abca1', 'Abcg1', 
          'Tgfb1','Pparg','Arg1', 'Retnla', 'Igf1', 'Lyve1', 'Icosl', 'Ecm1', 'S100a4',
          'Il1b','Tnf','Csf3','Cxcl10','Ccl3', 'Ccl4', 'Ccl5', 'Nfkbia', 'Nlrp3','Nos2','Arg2')

for (i in unique(df_plot$cellName)) {
  print(i)
  print(MyGeneV(markers,df_plot[df_plot$cellName==i,'names']))
  print("------------------------------")
}


## ------- 3. module score -------
moduleScore <- read.table("./module_score.csv", sep=',')
moduleScore_Mertk <- moduleScore[moduleScore$cell_type2 == "Mertk hi Macro",]

moduleScore_Mertk$Group <- factor(moduleScore_Mertk$Group, levels = c('WT','TKO'))

p1<-ggplot(moduleScore_Mertk, aes(Group, efferocytosis_mouse)) +
  geom_boxplot(na.rm = T, outlier.shape = NA, colour = "black") +
  # geom_jitter(width = 0.2, alpha = 1, size = 2, color = "grey56") +
  ggpubr::stat_compare_means(
    comparisons = list( c("WT", "TKO")),
    method = "wilcox.test",  # 或 "t.test", "anova", "kruskal.test" 等
    label = "p.format",      # 显示 p 值格式
    label.x.npc = "center",  # 水平位置
    label.y.npc = "top",     # 垂直位置
    vjust = -0.5             # 垂直调整
  ) +
  labs(title = 'Efferocytosis', y = 'Module score') +
  theme_wf2 +
  theme(axis.title.x.bottom = element_blank())

p3<-ggplot(moduleScore_Mertk, aes(Group, efferocytosis_mouse2)) +
  geom_boxplot(na.rm = T, outlier.shape = NA, colour = "black") +
  # geom_jitter(width = 0.2, alpha = 1, size = 2, color = "grey56") +
  ggpubr::stat_compare_means(
    comparisons = list( c("WT", "TKO")),
    method = "wilcox.test",  # 或 "t.test", "anova", "kruskal.test" 等
    label = "p.format",      # 显示 p 值格式
    label.x.npc = "center",  # 水平位置
    label.y.npc = "top",     # 垂直位置
    vjust = -0.5             # 垂直调整
  ) +
  labs(title = 'Efferocytosis', y = 'Module score') +
  theme_wf2 +
  theme(axis.title.x.bottom = element_blank())

p2<-ggplot(moduleScore_Mertk, aes(Group, IR_mouse)) +
  geom_boxplot(na.rm = T, outlier.shape = NA, colour = "black") +
  # geom_jitter(width = 0.2, alpha = 1, size = 2, color = "grey56") +
  ggpubr::stat_compare_means(
    comparisons = list( c("WT", "TKO")),
    method = "wilcox.test",  # 或 "t.test", "anova", "kruskal.test" 等
    label = "p.format",      # 显示 p 值格式
    label.x.npc = "center",  # 水平位置
    label.y.npc = "top",     # 垂直位置
    vjust = -0.5             # 垂直调整
  ) +
  labs(title = 'Inflammatory\nresponse', y = 'Module score') +
  theme_wf2 +
  theme(axis.title.x.bottom = element_blank())

pdf("./plot_plus_module_score.pdf",9,6)
p1+p2+p3
dev.off()

