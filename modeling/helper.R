

# Load necessary libraries for data manipulation and analysis
packages <- c("data.table", "magrittr")

for (pkg in packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
    library(pkg, character.only = TRUE)
  }
}

root_path <- "/home/luojiawei/inspire_benchmark_data/all_op_id"
operation_ <- fread("/home/luojiawei/inspire_benchmark_data/operation_.csv")
dim(operation_)

folders <- list.files(root_path)
read_data <- function(k) {
    op_id <- folders[k]
    res <- list()
    res[[1]] <- fread(file.path(root_path, op_id, "y_mat.csv"), header = T)
    res[[2]] <- fread(file.path(root_path, op_id, "y_mask.csv"), header = T)
    if(k %% 1000 == 0) print(k)
    return(res)
}

res_list <- lapply(1:length(folders), read_data)
y_mat <- do.call(rbind, lapply(res_list, function(x) x[[1]])) %>% as.data.frame
y_mask <- do.call(rbind, lapply(res_list, function(x) x[[2]])) %>% as.data.frame


for(i in 1:ncol(y_mat)){
    y_mat[y_mask[,i] == 0, i] <- NA
}

normalize <- function(x) {
  return(x / sum(x))
}

output_size_list <- apply(y_mat[,1:ncol(y_mat),drop=F], 2, function(x) length(unique(na.omit(x)))) %>%
                        ifelse(. > 2, ., 1)
type_list <- rep("cat", length(output_size_list))

weight_list_dym <- list()
for(i in 1:ncol(y_mat)){
    if(output_size_list[i] == 1) {
        m <- mean(y_mat[,i,drop=T],na.rm=T)
        m <- min(max(0.001, m),0.999)
        weight_list_dym[[names(y_mat)[i]]] <- normalize(c(1/(1-m), 1/m))
    } else{
        u <- factor(y_mat[,i,drop=T],levels=0:(output_size_list[i]-1))
        m <- prop.table(table(u))
        m <- pmin(pmax(0.001, m),0.999)
        m <- normalize(1/m) %>% as.vector
        weight_list_dym[[names(y_mat)[i]]] <- m
    }
}

for(i in 1:length(weight_list_dym)){
    cat(names(weight_list_dym)[i],"\n")
    cat(paste0(round(weight_list_dym[[i]],3),collapse = ","))
    cat("\n")
}

# mbp 
# 0.073,0.188,0.739
# hr 
# 0.113,0.458,0.43

