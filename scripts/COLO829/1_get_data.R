library(vcfR)
library(dplyr)
library(optparse)
library(patchwork)
library(tidyverse)

.libPaths('/orfeo/LTS/LADE/LT_storage/lvaleriani/R/x86_64-pc-linux-gnu-library/4.3')
library(Battenberg)

KMIN <- 3 
PHASING_GAMMA <- 3 
PHASING_KMIN <- 1 

setwd('/orfeo/scratch/area/lvaleriani/locate/scripts/COLO829/')
source('/orfeo/scratch/area/lvaleriani/COLO829/nf_analysis/utils.R')
out <- '/orfeo/cephfs/scratch/area/lvaleriani/locate/scripts/COLO829/out/data'

sample = 'COLO829'

out_dir <- file.path(out, sample)
dir.create(out_dir, recursive = T, showWarnings = F)

all_data <- lapply(1:22, FUN = function(chr){
  print(chr)
  if (sample == 'COLO829'){
    path_longphase <- paste0('/orfeo/scratch/area/lvaleriani/nf-locate/results_colo829/haplotagphase/',sample,'/',sample,'_chr',chr,'_phased.vcf.gz')
    path_shapeit <- paste0('/orfeo/scratch/area/lvaleriani/nf-locate/results_colo829/shapeit4/',sample,'/shapeit_',sample,'-chr',chr,'.vcf')
    path_normal <- paste0('/orfeo/scratch/area/lvaleriani/nf-locate/results_colo829/longphase/COLO829/Normal/COLO829_chr',chr,'_snp.vcf')
  } else{
    path_longphase <- paste0('/orfeo/scratch/area/lvaleriani/nf-locate/results_colo829_contamination/haplotagphase/',sample,'/',sample,'_chr',chr,'_phased.vcf.gz')
    path_shapeit <- paste0('/orfeo/scratch/area/lvaleriani/nf-locate/results_colo829_contamination/shapeit4/',sample,'/shapeit_',sample,'-chr',chr,'.vcf')
    path_normal <- paste0('/orfeo/scratch/area/lvaleriani/nf-locate/results_colo829_contamination/longphase/',sample,'/Normal/',sample,'_chr',chr,'_snp.vcf')
  }
  
  normal <- read_vcf(path_normal) %>%
    filter(REF %in% c('A', 'G', 'C', 'T'),
           ALT %in% c('A', 'G', 'C', 'T')) %>%
    select(CHROM, POS, gt_DP, gt_GT) 
  
  longphase <- read_vcf(path_longphase) %>%
    filter(REF %in% c('A', 'G', 'C', 'T'),
           ALT %in% c('A', 'G', 'C', 'T')) %>%
    mutate(BAF_H1_LP = ifelse(H1 == 1, BAF, 1-BAF)) %>%
    mutate(BAF_H2_LP = ifelse(H2 == 1, BAF, 1-BAF))
  
  shapeit <- read_vcf_shapeit(path = path_shapeit) %>%
    filter(REF %in% c('A', 'G', 'C', 'T'),
           ALT %in% c('A', 'G', 'C', 'T'))
  
  join <- longphase %>%
    left_join(shapeit, by = join_by(CHROM, POS), suffix = c('_LP', '_SI')) %>%
    filter(FILTER_LP == 'PASS') %>%
    dplyr::mutate(BAF_H1_SI = ifelse(H1_SI == 1, BAF, 1-BAF)) %>%
    dplyr::mutate(BAF_H2_SI = ifelse(H2_SI == 1, BAF, 1-BAF)) %>%
    filter(QUAL_LP >= 10)  %>%
    left_join(normal, by = join_by(CHROM, POS),suffix = c('_T', '_N') ) 
  saveRDS(object = join, file = paste0(out_dir, '/chr_', chr, '_join.rds'))
  
  join <- join %>% filter(gt_DP_T>15) %>% filter(QUAL_LP>15)
  data <- join %>% filter(!is.na(BAF_H1_SI), !is.na(BAF_H2_SI))
  sdev <- Battenberg:::getMad(ifelse(data$BAF_H1_SI < 0.5, data$BAF_H1_SI, 1 - data$BAF_H1_SI), k = 25)
  if (sdev < 0.002) {
    sdev <- 0.002
  }
  
  hap_segs <- Battenberg:::selectFastPcf(data$BAF_H1_SI, kmin = PHASING_KMIN, gamma = PHASING_GAMMA*sdev, yest = T)
  data$segment_H1 = hap_segs$yhat
  data$phased_H1 <- ifelse(data$segment_H1 < 0.5, 1 - data$BAF_H1_SI, data$BAF_H1_SI)
  data$phased_H2 <- ifelse(data$segment_H1 < 0.5, 1 - data$BAF_H2_SI, data$BAF_H2_SI)
  saveRDS(object = data, file = paste0(out_dir, '/chr_', chr, '_final.rds'))
  
  return(data) 
}) %>% bind_rows()
saveRDS(all_data, file =  paste0(out_dir, '/final.rds'))

data <- readRDS(paste0(out_dir, '/final.rds'))
dp_normal <- mean(data$gt_DP_N, na.rm = T)
dp_tumor <- mean(data$gt_DP_T, na.rm = T)
ratio <- dp_normal/dp_tumor

data <- data %>% mutate(DR = (gt_DP_T/gt_DP_N)*ratio)
data <- data %>%
  filter(gt_DP_T < 100) %>%
  filter(gt_DP_N < 100) %>%
  filter(gt_DP_N > 20) %>%
  filter(!is.na(phased_H1)) %>% 
  filter(gt_GT %in% c('0/1', '0|1', '1|0')) %>% 
  select(CHROM, POS, gt_DP_T, gt_GT_LP, gt_GT_SI, BAF, gt_DP_N, phased_H1, phased_H2, DR)


plt <- lapply(paste0('chr',1:22), FUN = function(c){
  p <- data %>%
  filter(CHROM == c) %>%
  ggplot() +
  geom_point(aes(x = POS, y = phased_H1), size = .1, col = 'goldenrod') +
  #geom_point(aes(x = POS, y = phased_H2), size = .1, col = 'steelblue')  +
  ylim(0,1) +
  ggtitle(c) +
  data %>%
  filter(CHROM == c) %>%
  ggplot() +
  geom_point(aes(x = POS, y = DR), size = .1) +
  ylim(0,3) + plot_layout(nrow = 2) & theme_minimal()
  return(p)
})
