

cat("start data generation\n")

d_lab <- read_excel("/home/luojiawei/pengxiran_project/inspire_new_folder/var_dict.xlsx", sheet=1, col_names = T)
d_vit <- read_excel("/home/luojiawei/pengxiran_project/inspire_new_folder/var_dict.xlsx", sheet=2, col_names = T)
d_static <- read_excel("/home/luojiawei/pengxiran_project/inspire_new_folder/var_dict.xlsx", sheet=3, col_names = T)
d_ward_vit <- read_excel("/home/luojiawei/pengxiran_project/inspire_new_folder/var_dict.xlsx", sheet=4, col_names = T)

d_lab[1:2,]
d_vit[1:2, ]
d_static[1:2, ]
d_ward_vit[1:2, ]

labs <- fread(file.path(inspire_path, "labs.csv"))
labs_ <- labs[labs$subject_id %in% operation_$subject_id[operation_$dataset==1]]
dim(labs_)

vitals <- fread(file.path(inspire_path, "vitals.csv"))
vitals_ <- vitals[vitals$subject_id %in% operation_$subject_id[operation_$dataset==1]]
dim(vitals_)

ward_vitals <- fread(file.path(inspire_path, "ward_vitals.csv"))
ward_vitals_ <- ward_vitals[ward_vitals$subject_id %in% operation_$subject_id[operation_$dataset==1]]
dim(ward_vitals_)

stat_lab <- get_stat_long(labs_, 
                          d_lab$itemid, 
                          d_lab$value_type,
                          "item_name",
                          "value",
                          d_lab$cont,
                          d_lab)

stat_vit <- get_stat_long(vitals_, 
                          d_vit$itemid, 
                          d_vit$value_type,
                          "item_name",
                          "value",
                          d_vit$cont,
                          d_vit)

stat_ward_vit <- get_stat_long(ward_vitals_, 
                          d_ward_vit$itemid, 
                          d_ward_vit$value_type,
                          "item_name",
                          "value",
                          d_ward_vit$cont,
                          d_ward_vit,
                          sep = "\\|")

stat_static <- get_stat_wide(operation_[dataset==1,d_static$itemid,with=F],
                              d_static$itemid,
                              d_static$value_type,
                              d_static$cont,
                              d_static,
                              sep = "\\|")


imp <- mice(operation_[operation_$dataset==1,d_static$itemid,with=F], method = "pmm", seed = 123, m=1, maxit = 10, printFlag = F)
imp_data <- complete(imp)
operation_[operation_$dataset==1,d_static$itemid] <- imp_data


imp <- mice(operation_[operation_$dataset==2,d_static$itemid,with=F], method = "pmm", seed = 123, m=1, maxit = 10, printFlag = F)
imp_data <- complete(imp)
operation_[operation_$dataset==2,d_static$itemid] <- imp_data

imp <- mice(operation_[operation_$dataset==3,d_static$itemid,with=F], method = "pmm", seed = 123, m=1, maxit = 10, printFlag = F)
imp_data <- complete(imp)
operation_[operation_$dataset==3,d_static$itemid] <- imp_data


vitals <- vitals[order(vitals_$op_id),]
labs <- labs[order(labs$subject_id),]
ward_vitals <- ward_vitals[order(ward_vitals$subject_id),]

vitals_index <- data.table(
  op_id = unique(vitals$op_id),
  start_idx = match(unique(vitals$op_id), vitals$op_id),
  end_idx = c(match(unique(vitals$op_id), vitals$op_id)[-1] - 1, nrow(vitals))
)
setkey(vitals_index, op_id)

labs_index <- data.table(
  subject_id = unique(labs$subject_id),
  start_idx = match(unique(labs$subject_id), labs$subject_id),
  end_idx = c(match(unique(labs$subject_id), labs$subject_id)[-1] - 1, nrow(labs))
)
setkey(labs_index, subject_id)

ward_vitals_index <- data.table(
  subject_id = unique(ward_vitals$subject_id),
  start_idx = match(unique(ward_vitals$subject_id), ward_vitals$subject_id),
  end_idx = c(match(unique(ward_vitals$subject_id), ward_vitals$subject_id)[-1] - 1, nrow(ward_vitals))
)
setkey(ward_vitals_index, subject_id) 


get_1opid_data <- function(k) {
#   k <- 90
  cur_opid <- operation_$op_id[k]
  cur_sid <- operation_$subject_id[k]
  cur_hid <- operation_$hadm_id[k]
  
  operation_row <- operation_[which(operation_$op_id == cur_opid),]
  
  t0 <- operation_row$admission_time
  t1 <- operation_row$discharge_time
  t_in <- operation_row$orin_time
  t_out <- operation_row$orout_time
  t_start <- operation_row$opstart_time
  t_end <- operation_row$opend_time

  vit_idx <- vitals_index[.(cur_opid), on = "op_id"]
  lab_idx <- labs_index[.(cur_sid), on = "subject_id"]
  ward_idx <- ward_vitals_index[.(cur_sid), on = "subject_id"]
  
  if(nrow(vit_idx) > 0 && !is.na(vit_idx$start_idx) && !is.na(vit_idx$end_idx)) {
    vit_k <- vitals[vit_idx$start_idx:vit_idx$end_idx, 2:5, drop=F]
  } else {
    warning(sprintf("找不到vitals索引或索引为NA，cur_opid: %s", cur_opid))
    vit_k <- vitals[0, drop=F]
  }

  if(nrow(lab_idx) > 0 && !is.na(lab_idx$start_idx) && !is.na(lab_idx$end_idx)) {
    lab_k <- labs[lab_idx$start_idx:lab_idx$end_idx,]
  } else {
    warning(sprintf("找不到labs索引或索引为NA，cur_sid: %s", cur_sid))
    lab_k <- labs[0,]
  }

  if(nrow(ward_idx) > 0 && !is.na(ward_idx$start_idx) && !is.na(ward_idx$end_idx)) {
    ward_vit_k <- ward_vitals[ward_idx$start_idx:ward_idx$end_idx,]
  } else {
    warning(sprintf("找不到ward_vitals索引或索引为NA，cur_sid: %s", cur_sid))
    ward_vit_k <- ward_vitals[0,]
  }

  t_lab <- sort(unique(lab_k$chart_time))
  t_lab <- t_lab[t_lab >= t0 & t_lab <= t_in]
  if(length(t_lab) == 0){
    t_lab <- c(t_in)
  }
  lab_k1 <- resample_long(lab_k,
                              d_lab$itemid,
                              d_lab$value_type,
                              d_lab$agg_f,
                              t_lab,
                              "chart_time",
                              NULL,
                              "item_name",
                              "value",
                              12 * 60,
                              direction = "left",
                              keepNArow = F,
                              keep_first = T)
  lab_k1 <- lab_k1[,c(1,3:ncol(lab_k1)),drop=F]
  lab_raw <- lab_k1
  mask_lab <- get_mask(lab_k1, colnames(lab_k1)[2:ncol(lab_k1)], "time")
  lab_k1 <- fill(lab_k1, 2:ncol(lab_k1), get_type(stat_lab), d_lab$fill1, d_lab$fill2, stat_lab, 1)
  if(nrow(lab_k1) == 1){
    lab_k1$dt <- c(0)
  } else{
    lab_k1$dt <- c(0, diff(lab_k1$time))
  }
  lab_k1$dt <- lab_k1$dt / 60
  lab_k1[,2:ncol(lab_k1)] <- lab_k1[,2:ncol(lab_k1)] %>% lapply(., as.numeric)

  t_ward_vit <- sort(unique(ward_vit_k$chart_time))
  t_ward_vit <- t_ward_vit[t_ward_vit >= t0 & t_ward_vit <= t_in]
  if(length(t_ward_vit) == 0){
    t_ward_vit <- c(t_in)
  }
  ward_vit_k1 <- resample_long(ward_vit_k,
                              d_ward_vit$itemid,
                              d_ward_vit$value_type,
                              d_ward_vit$agg_f,
                              t_ward_vit,
                              "chart_time",
                              NULL,
                              "item_name",
                              "value",
                              5,
                              direction = "both",
                              keepNArow = F,
                              keep_first = F)
  
  ward_vit_k1 <- ward_vit_k1[,c(1,3:ncol(ward_vit_k1)),drop=F]
  ward_vit_raw <- ward_vit_k1
  ward_vit_k1$nibp_mbp <- (ward_vit_k1$nibp_sbp + 2 * ward_vit_k1$nibp_dbp) / 3
  mask_ward_vit <- get_mask(ward_vit_k1, colnames(ward_vit_k1)[2:ncol(ward_vit_k1)], "time")
  ward_vit_k1 <- fill(ward_vit_k1, 2:ncol(ward_vit_k1), get_type(stat_ward_vit), d_ward_vit$fill1, d_ward_vit$fill2, stat_ward_vit, 1)
  ward_vit_k1 <- fill_last_values(ward_vit_k1, mask_ward_vit, colnames(ward_vit_k1)[2:ncol(ward_vit_k1)], "time", d_ward_vit)
  ward_vit_k1 <- to_onehot(ward_vit_k1, 2:ncol(ward_vit_k1), "time", get_type(stat_ward_vit), stat_ward_vit)
  if(nrow(ward_vit_k1) == 1){
    ward_vit_k1$dt <- c(0)
  } else{
    ward_vit_k1$dt <- c(0, diff(ward_vit_k1$time))
  }
  ward_vit_k1$dt <- ward_vit_k1$dt / 60
  ward_vit_k1[,2:ncol(ward_vit_k1)] <- ward_vit_k1[,2:ncol(ward_vit_k1)] %>% lapply(., as.numeric)
  
  
  t_vit <- sort(unique(vit_k$chart_time))
  t_vit <- seq(t_in, t_out, 5)
  vit_k1 <- resample_long(vit_k,
                              d_vit$itemid,
                              d_vit$value_type,
                              d_vit$agg_f,
                              t_vit,
                              "chart_time",
                              NULL,
                              "item_name",
                              "value",
                              5,
                              direction = "both",
                              keepNArow = T,
                              keep_first = F)
  
  vit_k1 <- vit_k1[,c(1,3:ncol(vit_k1)),drop=F]
  vit_raw <- vit_k1
  vit_k1$nibp_sbp <- ifelse(is.na(vit_k1$nibp_sbp), vit_k1$art_sbp, vit_k1$nibp_sbp)
  vit_k1$nibp_dbp <- ifelse(is.na(vit_k1$nibp_dbp), vit_k1$art_dbp, vit_k1$nibp_dbp)
  vit_k1$art_mbp <- (vit_k1$art_sbp + 2 * vit_k1$art_dbp) / 3
  vit_k1$nibp_mbp <- (vit_k1$nibp_sbp + 2 * vit_k1$nibp_dbp) / 3
  X_vit <- vit_k1[vit_k1$time >= t_in & vit_k1$time <= t_out,,drop=F]
  mask_vit <- get_mask(vit_k1, colnames(vit_k1)[2:ncol(vit_k1)], "time")
  vit_k1 <- fill(vit_k1, 2:ncol(vit_k1), get_type(stat_vit), d_vit$fill1, d_vit$fill2, stat_vit, 1)
  vit_k1[,2:ncol(vit_k1)] <- vit_k1[,2:ncol(vit_k1)] %>% lapply(., as.numeric)

  y_mat <- list()
  u <- as.numeric(ward_vit_k1$nibp_mbp)
  u1 <- as.numeric(X_vit$nibp_mbp)
  m <- mean(u, na.rm=T)
  u1 <- ifelse(u1 < 0.8*m, 1, ifelse(u1 < 1.2*m, 0, 2))
  y_mat[[1]] <- cbind("mbp" = u1)

  u <- as.numeric(ward_vit_k1$hr)
  u1 <- as.numeric(X_vit$hr)
  m <- mean(u, na.rm=T)
  u1 <- ifelse(u1 < 0.8*m, 1, ifelse(u1 < 1.2*m, 0, 2))
  y_mat[[2]] <- cbind("hr" = u1)

  y_mat <- do.call(cbind, y_mat)
  y_mask <- rbind(apply(y_mat, 2, function(x) ifelse(is.na(x), 0, 1)))
  y_mask <- y_mask %>% convert_to_numeric_df()
  y_mat <- rbind(apply(y_mat, 2, function(x) ifelse(is.na(x), 0, x)))
  y_mat <- y_mat %>% convert_to_numeric_df()

  x_s <- operation_row[,d_static$itemid,with=F] %>% as.data.frame()
  x_s <- to_onehot(x_s, 1:ncol(x_s), NULL, get_type(stat_static), stat_static)
  
  names(operation_row)
  index <- c("death_30d","have_icu","have_aki","have_ali","postop_lung_complications","postop_cardiac_complications","postop_stroke")
  y_static <- operation_row[,..index]
  y_mask1 <- rbind(apply(y_static, 2, function(x) ifelse(is.na(x), 0, 1)))
  y_mask1 <- y_mask1 %>% convert_to_numeric_df()
  y_static <- rbind(apply(y_static, 2, function(x) ifelse(is.na(x), 0, x)))
  y_static <- y_static %>% convert_to_numeric_df()

  t_list <- data.frame("time"=seq(t_in, t_out, 5))
  
  return(list(lab_k1, mask_lab, 
              vit_k1, mask_vit, 
              ward_vit_k1, mask_ward_vit, 
              t_list, x_s, y_mat, y_mask, y_static, y_mask1,
              lab_raw, vit_raw, ward_vit_raw))
}

process_data <- function(k, root_path) {
    id_k<-operation_$op_id[k]
    folder_path<-file.path(root_path, id_k)
    create_dir(folder_path, F)
    
    datas <- get_1opid_data(k)

    fwrite(datas[[1]], file=file.path(folder_path, "lab.csv"), row.names=F)
    fwrite(datas[[2]], file=file.path(folder_path, "mask_lab.csv"), row.names=F)

    fwrite(datas[[3]], file=file.path(folder_path, "vit.csv"), row.names=F)
    fwrite(datas[[4]], file=file.path(folder_path, "mask_vit.csv"), row.names=F)

    fwrite(datas[[5]], file=file.path(folder_path, "ward_vit.csv"), row.names=F)
    fwrite(datas[[6]], file=file.path(folder_path, "mask_ward_vit.csv"), row.names=F)

    fwrite(datas[[7]], file=file.path(folder_path, "t_list.csv"), row.names=F)
    fwrite(datas[[8]], file=file.path(folder_path, "x_s.csv"), row.names=F)
    fwrite(datas[[9]], file=file.path(folder_path, "y_mat.csv"), row.names=F)
    fwrite(datas[[10]], file=file.path(folder_path, "y_mask.csv"), row.names=F)
    fwrite(datas[[11]], file=file.path(folder_path, "y_static.csv"), row.names=F)
    fwrite(datas[[12]], file=file.path(folder_path, "y_mask1.csv"), row.names=F)

    fwrite(datas[[13]], file=file.path(folder_path, "lab_raw.csv"), row.names=F)
    fwrite(datas[[14]], file=file.path(folder_path, "vit_raw.csv"), row.names=F)
    fwrite(datas[[15]], file=file.path(folder_path, "ward_vit_raw.csv"), row.names=F)
}


create_dir(root_path, T)

chunk_size <- 3000
num_rows <- nrow(operation_)
num_chunks <- ceiling(num_rows / chunk_size)

results <- list()

for (i in 1:num_chunks) {
  start_index <- (i - 1) * chunk_size + 1
  end_index <- min(i * chunk_size, num_rows)
  
  results[[i]] <- mclapply(start_index:end_index, 
                    function(x) {
                      result <- tryCatch({
                        process_data(x, root_path = root_path)
                      }, error = function(e) {
                        print(e)
                        print(x)
                      })
                      if(x %% 1000 == 0) print(x)
                      
                      return(result)
                    }, mc.cores = detectCores())
  gc()
}


z_param_vit <- init_z_param(d_vit$itemid, d_vit, stat_vit)
z_param_vit<- data.frame(
  var = names(z_param_vit),
  mean = sapply(z_param_vit, function(x) x$mean),
  sd = sapply(z_param_vit, function(x) x$sd)
)

z_param_static <- init_z_param(d_static$itemid, d_static, stat_static)
z_param_static<- data.frame(
  var = names(z_param_static),
  mean = sapply(z_param_static, function(x) x$mean),
  sd = sapply(z_param_static, function(x) x$sd)
)

z_param_lab <- init_z_param(d_lab$itemid, d_lab, stat_lab)
z_param_lab <- data.frame(
  var = names(z_param_lab),
  mean = sapply(z_param_lab, function(x) x$mean),
  sd = sapply(z_param_lab, function(x) x$sd)
)
# 添加 dt 变量
z_param_lab <- rbind(z_param_lab, data.frame(var = "dt", mean = 0, sd = 1))

z_param_ward_vit <- init_z_param(d_ward_vit$itemid, d_ward_vit, stat_ward_vit)
z_param_ward_vit <- data.frame(
  var = names(z_param_ward_vit),
  mean = sapply(z_param_ward_vit, function(x) x$mean),
  sd = sapply(z_param_ward_vit, function(x) x$sd)
)
# 添加 dt 变量
z_param_ward_vit <- rbind(z_param_ward_vit, data.frame(var = "dt", mean = 0, sd = 1))

write.csv(z_param_vit, file = "./param_folder/z_param_vit.csv", row.names = FALSE)
write.csv(z_param_static, file = "./param_folder/z_param_static.csv", row.names = FALSE)
write.csv(z_param_lab, file = "./param_folder/z_param_lab.csv", row.names = FALSE)
write.csv(z_param_ward_vit, file = "./param_folder/z_param_ward_vit.csv", row.names = FALSE)

cat("finish data generation\n")