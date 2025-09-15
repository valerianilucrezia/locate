library(tidyverse)
library(patchwork)
base = '/orfeo/scratch/area/lvaleriani/utils_locate/simulations_rRACES/out/clonal'
sims = paste0('sim_', 1:60)

#vaf_False_ploidy_True_bps_False

all_df <- list()
all_plt <- list()

S = c('vaf_True_ploidy_True_bps_False', 'vaf_True_ploidy_False_bps_False')
#suffix = paste('vaf',vaf, 'ploidy', ploidy, 'bps', bps, sep = '_') 

for (suffix in S){
  print(suffix)
  tmp <- strsplit(suffix, "_")[[1]]
  vaf <- tmp[2]
  
  df <- tibble()
  plt <- list()
  for(sim in sims){
    path = file.path(base, sim)
    combinations = list.dirs(path, recursive = F)
    for (comb in combinations){
      if (file.exists(file.path(comb, suffix, paste0('cna.csv')))){
        sim_cna <- read.csv(file.path(comb, paste0('mirr_smooth_snv.csv'))) %>% 
          select(pos, major, minor, CN) %>% 
          arrange(pos) %>% 
          mutate(POS = 1:n())
        
        inf_cna <- read.csv(file.path(comb, suffix, paste0('cna.csv')))
        params = read.csv(file.path(comb, suffix, paste0('params.csv'))) 
        
        metrics = read.csv(file.path(comb, suffix, paste0('summary.csv'))) %>% 
          tibble() %>% 
          tidyr::separate(sample, sep = '_', into = c('sim_name', 'sim', 'cov_name', 'cov', 'pur_name', 'pur')) %>% 
          select(-sim_name, -cov_name, -pur_name) %>% 
          mutate(vaf = vaf, 
                 ploidy = ploidy, 
                 bps = bps, 
                 delta_purity = params$purity - params$inf_purity,
                 delta_ploidy = params$ploidy - params$inf_ploidy,
                 true_ploidy = params$ploidy) 
        df <- bind_rows(df, metrics)
        
        if (metrics$allelic_accuracy < 0.7){
          plt[[paste(sim, metrics$cov,metrics$pur, sep = '_')]] <- ggplot() +
              geom_point(data = sim_cna, aes(x = POS,  y = major+0.03), col = 'deepskyblue4') +
              geom_point(data = sim_cna, aes(x = POS,  y = minor-0.03), col = 'indianred3') +
              ggtitle(paste(metrics$sim, metrics$cov, metrics$pur, sep = '_'), 
                      subtitle = paste0('Inf purity = ', round(params$inf_purity,2), '\nPloidy = ', round(params$ploidy,2), '\nInf ploidy =', round(params$inf_ploidy,2)))+
              ylab('Simulated CN') +
          ggplot() +
              geom_point(data = inf_cna, aes(x = pos,  y = CN_Major+0.03), col = 'deepskyblue4') +
              geom_point(data = inf_cna, aes(x = pos,  y = CN_minor-0.03), col = 'indianred3') +
              ylab('Inferred CN') +
              plot_layout(nrow = 2) & theme_minimal()
        }
      }
    }
  }
  
  all_df[[suffix]] <- df
  all_plt[[suffix]] <- plt
}


plot_stat_cn <- lapply(S, FUN = function(suffix){
  all <- all_df[[suffix]]  %>% 
    pivot_longer(cols = c(allelic_accuracy)) %>% 
    ggplot() +
    geom_boxplot(aes(x = name, y = value, fill = cov)) +
    xlab('metric') +
    facet_grid(.~pur) + 
    ggtitle(suffix) + 
    theme_minimal() 
})
wrap_plots(plot_stat_cn)


plot_stat_params <- lapply(S, FUN = function(suffix){
  all <- all_df[[suffix]]  %>% 
    ggplot() +
    geom_boxplot(aes(x = pur, y = delta_purity, fill = cov)) + 
    ylab('true purity -  inf purity') +
    xlab('true purity') +
    ggtitle(suffix) + 
    theme_minimal()  +
    ylim(-0.5, 1) + 
    
    all_df[[suffix]]  %>% 
    ggplot() +
    geom_boxplot(aes(x = pur, y = delta_ploidy, fill = cov)) + 
    ylab('true ploidy -  inf ploidy') +
    xlab('true purity') +
    ggtitle(suffix) + 
    theme_minimal() 
})
wrap_plots(plot_stat_params)
