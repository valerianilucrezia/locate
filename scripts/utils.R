get_germline <- function(sample, base){
  all_data <- lapply(1:22, FUN = function(chr){
    print(chr)
    path_longphase <- paste0(base, '/haplotagphase/',sample,'/',sample,'_chr',chr,'_phased.vcf.gz')
    path_shapeit <- paste0(base, '/',sample,'/shapeit_',sample,'-chr',chr,'.vcf')
    path_normal <- paste0(base, '/longphase/', sample, '/Normal/', sample, '_chr',chr,'_snp.vcf')

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
  
  dp_normal <- mean(all_data$gt_DP_N, na.rm = T)
  dp_tumor <- mean(all_data$gt_DP_T, na.rm = T)
  ratio <- dp_normal/dp_tumor
  
  data <- all_data %>% mutate(DR = (gt_DP_T/gt_DP_N)*ratio)
  data <- all_data %>%
    filter(gt_DP_T < 100) %>%
    filter(gt_DP_N < 100) %>%
    filter(gt_DP_N > 20) %>%
    filter(!is.na(phased_H1)) %>% 
    filter(gt_GT %in% c('0/1', '0|1', '1|0')) %>% 
    select(CHROM, POS, gt_DP_T, gt_GT_LP, gt_GT_SI, BAF, gt_DP_N, phased_H1, phased_H2, DR)
  saveRDS(all_data, file =  paste0(out_dir, '/filter_final.rds'))
}


get_somatic <- function(sample, base = '/orfeo/cephfs/scratch/area/lvaleriani/nf-locate/results_colo829'){
  path_somatic <- paste0(base, '/variant_calling/clairS/',sample,'/',sample,'_variants.vcf.gz')
  
  somatic <- read.vcfR(path_somatic) %>% vcfR::vcfR2tidy()
  somatic_fix <- somatic$fix %>% 
    select(ChromKey, CHROM, POS, REF, ALT, FILTER, H, QUAL) %>% 
    filter(FILTER == 'PASS')
  somatic_gt <- somatic$gt %>% 
    select(ChromKey, POS, gt_GT, gt_DP, gt_AF, gt_AD) 
  
  somatic_data <- left_join(somatic_fix, somatic_gt, by = join_by(ChromKey, POS)) %>% 
    filter(gt_DP >= 10)  %>% 
    filter(gt_GT != '1/1')
  
  
}

absolute_to_relative_coordinates <- function(muts, reference = CNAqc::chr_coordinates_GRCh38){
  vfrom = reference$from
  names(vfrom) = reference$chr
  
  muts %>%
    mutate(
      POS = POS + vfrom[CHROM])
}

# smooting ####
data_smoothing <- function(sp, res_path, bin_size = 50000, save = TRUE){
  smooth_snp <- sp %>%
    rename(pos = from.tumour) %>%
    mutate(group = ceiling(pos/ bin_size)) %>% 
    group_by(group, cna_id.tumour) %>% 
    summarize(mean_BAF = mean(VAF.tumour),
              median_BAF = median(VAF.tumour),
              mean_DR = mean(DR),
              median_DR = median(DR), 
              nSNP = n_distinct(pos), 
              minSNP = min(pos),
              maxSNP = max(pos),
    )  
  
  
  if (save == TRUE){
    saveRDS(smooth_snp, paste0(res_path, 'smooth_snp.RDS') )
    return(smooth_snp)
  } else if( save == FALSE){
    return(smooth_snp)
  }
}


vaf_smoothing <- function(sv, sp, res_path, save = TRUE, vaf_th = 0.15, wd = 20000){
  snv <- sv %>% filter(VAF >= vaf_th) %>% rename(pos = from) %>% rename(vaf = VAF)
  snp <- sp %>%  rename(pos = from.tumour)
  
  smooth_vaf <- tibble() 
  for (i in seq(1, nrow(snv))){
    tmp <- snv[i,]
    mean_baf <- snp %>% filter(pos > tmp$pos - wd, pos < tmp$pos + wd) %>% pull(VAF.tumour) %>% mean()
    mean_dr <- snp %>% filter(pos > tmp$pos - wd, pos < tmp$pos + wd) %>% pull(DR) %>% mean()
    median_baf <- snp %>% filter(pos > tmp$pos - wd, pos < tmp$pos + wd) %>% pull(VAF.tumour) %>% median()
    median_dr <- snp %>% filter(pos > tmp$pos - wd, pos < tmp$pos + wd) %>% pull(DR) %>% median()
    tmp$mean_dr = mean_dr
    tmp$mean_baf = mean_baf
    tmp$median_dr = median_dr
    tmp$median_baf = median_baf
    smooth_vaf =  bind_rows(smooth_vaf, tmp)
  }
  
  smooth_vaf <- smooth_vaf %>% arrange(desc(pos))
  smooth_vaf <- smooth_vaf %>% filter(!is.na(mean_baf))
  
  
  if (save == TRUE){
    saveRDS(smooth_vaf, paste0(res_path, 'smooth_vaf.RDS'))
    return(smooth_vaf)
  } else if (save == FALSE){
    return(smooth_vaf)
  }
}

mirro_and_smoothing <- function(sv, sp, res_path, save = TRUE, vaf_th = 0.15, wd = 20000){
  snv <- sv %>% rename(pos = from) %>% rename(vaf = VAF)
  snp <- sp %>%  rename(pos = from.tumour) %>% 
    mutate(new_baf = ifelse(VAF.tumour > 0.6, 1-VAF.tumour, VAF.tumour))
  
  smooth_vaf <- tibble() 
  for (i in seq(1, nrow(snv))){
    tmp <- snv[i,]
    mean_baf <- snp %>% filter(pos > tmp$pos - wd, pos < tmp$pos + wd) %>% pull(new_baf) %>% mean()
    mean_dr <- snp %>% filter(pos > tmp$pos - wd, pos < tmp$pos + wd) %>% pull(DR) %>% mean()
    median_baf <- snp %>% filter(pos > tmp$pos - wd, pos < tmp$pos + wd) %>% pull(new_baf) %>% median()
    median_dr <- snp %>% filter(pos > tmp$pos - wd, pos < tmp$pos + wd) %>% pull(DR) %>% median()
    mean_dp <- snp %>% filter(pos > tmp$pos - wd, pos < tmp$pos + wd) %>% pull(DP.tumour) %>% mean()
    median_dp <- snp %>% filter(pos > tmp$pos - wd, pos < tmp$pos + wd) %>% pull(DP.tumour) %>% median()
    tmp$mean_dr = mean_dr
    tmp$mean_baf = mean_baf
    tmp$median_dr = median_dr
    tmp$median_baf = median_baf
    tmp$mean_dp = mean_dp
    tmp$median_dp = median_dp
    smooth_vaf =  bind_rows(smooth_vaf, tmp)
  }
  
  smooth_vaf <- smooth_vaf %>% arrange(desc(pos))
  smooth_vaf <- smooth_vaf %>% filter(!is.na(mean_baf))
  
  
  if (save == TRUE){
    saveRDS(smooth_vaf, paste0(res_path, 'mirr_smooth_vaf.RDS'))
    write.csv(smooth_vaf, paste0(res_path, 'mirr_smooth_vaf.csv'), quote = F, row.names = F)
    return(smooth_vaf)
  } else if (save == FALSE){
    return(smooth_vaf)
  }
}

