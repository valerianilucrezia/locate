library(tidyverse)
library(vcfR)
library(patchwork)
setwd('/orfeo/cephfs/scratch/area/lvaleriani/locate/scripts')
source('utils.R')

.libPaths('/orfeo/LTS/LADE/LT_storage/lvaleriani/R/x86_64-pc-linux-gnu-library/4.3')
library(Battenberg)
KMIN <- 3 
PHASING_GAMMA <- 3 
PHASING_KMIN <- 1 

base = "/orfeo/cephfs/scratch/area/lvaleriani/nf-locate/results_colo829"
sample = "COLO829"

# germline
germline <- readRDS('/orfeo/cephfs/scratch/area/lvaleriani/locate/scripts/COLO829/out/data/COLO829/final.rds')
dp_normal <- mean(germline$gt_DP_N, na.rm = T)
dp_tumor <- mean(germline$gt_DP_T, na.rm = T)
ratio <- dp_normal/dp_tumor

germline_data <- germline %>% mutate(DR = (gt_DP_T/gt_DP_N)*ratio)
germline_data <- germline_data %>%
  filter(gt_DP_T < 100) %>%
  filter(gt_DP_N < 100) %>%
  filter(gt_DP_N > 20) %>%
  filter(!is.na(phased_H1)) %>% 
  filter(gt_GT %in% c('0/1', '0|1', '1|0')) %>% 
  select(CHROM, POS, gt_DP_T, gt_GT_LP, gt_GT_SI, BAF, gt_DP_N, phased_H1, phased_H2, DR)

# somatic
somatic_data <- get_somatic(sample, base)
plt <- somatic_data %>% 
  ggplot() +
  geom_histogram(aes(x = gt_AF), binwidth = 0.01) +
  facet_wrap(.~CHROM)

#smooth
bin_size = 30000
smooth <- absolute_to_relative_coordinates(germline_data) %>%
  group_by(CHROM) %>% 
  mutate(group = ceiling(POS/bin_size)) %>% 
  group_by(group, CHROM) %>% 
  summarize(mean_BAF = mean(phased_H1),
            median_BAF = median(phased_H1),
            mean_DP = mean(gt_DP_T),
            median_DP = median(gt_DP_T),
            mean_DR = mean(DR),
            median_DR = median(DR), 
            nSNP = n_distinct(POS), 
            minSNP = min(POS),
            maxSNP = max(POS),
  )

plt_1 <- smooth %>% 
  ggplot() +
  geom_point(aes(x = group, y = mean_BAF), size = .3) + 
  ylim(0,1) + 
  theme_minimal() + 
  
  smooth %>% 
  ggplot() +
  geom_point(aes(x = group, y = mean_DR), size = .3) + 
  ylim(0,3) + 
  theme_minimal() + 
  theme_minimal() + plot_layout(nrow = 2)

min_snps <- 10
smooth_filtered <- smooth %>% filter(nSNP >= min_snps) %>% ungroup() %>% mutate(group = 1:n()) 
plt_2 <- smooth_filtered %>% 
  ggplot() +
  geom_point(aes(x = group, y = mean_BAF, col = CHROM), size = .3) + 
  ylim(0,1) + 
  theme_minimal() + 
  
  smooth_filtered %>% 
  ggplot() +
  geom_point(aes(x = group, y = mean_DR, col = CHROM), size = .3) + 
  ylim(0,3) + 
  theme_minimal() + 
  theme_minimal() + plot_layout(nrow = 2)

smooth_filtered <- smooth_filtered %>% rename(pos = group)
dir.create(paste0('out/', sample, '/'), recursive = T, showWarnings = F)
saveRDS(object = smooth_filtered,  file = paste0('out/', sample, '/', sample, '_smooth.rds'))
write.csv(x = smooth_filtered, file = paste0('out/', sample, '/', sample, '_smooth.csv'), quote = F, row.names = F)
