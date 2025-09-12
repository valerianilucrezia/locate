library(tidyverse)
base = '/orfeo/scratch/area/lvaleriani/utils_locate/simulations_rRACES/test_dr/out/clonal'
sims = paste0('sim_', 1:60)

tol = 10
w_size = 5
thr = "1e-15"
mode = 'max'
suffix = paste(tol, mode, w_size, thr, sep = '_') 

df <- tibble()
for(sim in sims){
  path = file.path(base, sim)
  combinations = list.dirs(path, recursive = F)
  for (comb in combinations){
    if (file.exists(file.path(comb, paste0('bp_metrics_', suffix, '.csv')))){
      metrics = read.csv(file.path(comb, paste0('bp_metrics_', suffix, '.csv'))) %>% 
        tibble() %>% 
        tidyr::separate(run_id, sep = '_', into = c('sim_name', 'sim', 'cov_name', 'cov', 'pur_name', 'pur', 'mode', 'w_size', 'thr')) %>% 
        select(-X, -sim_name, -cov_name, -pur_name)
      df <- bind_rows(df, metrics)
    }
  }
}

all <- df %>% 
  pivot_longer(cols = c(precision, recall, f1)) %>% 
  ggplot() +
  geom_boxplot(aes(x = name, y = value, fill = cov)) +
  xlab('metric') +
  facet_grid(.~pur) + 
  ggtitle('Tolerance = 10bp') + 
  theme_minimal() 
all
