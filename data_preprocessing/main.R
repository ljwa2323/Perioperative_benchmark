
# This script is designed to load and preprocess the INSPIRE dataset for perioperative medicine research.
# It includes steps for data manipulation, handling missing data, and calculating various scores and outcomes.
# The script utilizes multiple R libraries to efficiently manage and analyze the dataset, ensuring reproducibility and accuracy in the results.

rm(list = ls()); gc()

# Load necessary libraries for data manipulation and analysis
packages <- c("readxl", "data.table", "dplyr", "magrittr", "lubridate", "stringr", "mice", "parallel", "pbapply", "jsonlite")

for (pkg in packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
    library(pkg, character.only = TRUE)
  }
}

source("data_preprocessing/EMR_LIP.R")

# Set the root path for the dataset, ensuring that all CSV files in INSPIRE are located in this path
inspire_path <- "/home/luojiawei/pengxiran_project_data/inspire-a-publicly-available-research-dataset-for-perioperative-medicine-1.2"

source("data_preprocessing/0_operation.R")
source("data_preprocessing/1_comorbidities.R") # This may take about 10 minutes to run
source("data_preprocessing/2_outcome.R") # This may take about 30 minutes to run
source("data_preprocessing/3_scores.R") # This may take about 10 minutes to run

operation[1:2,]
summary(operation)

fwrite(operation, "/home/luojiawei/inspire_benchmark_data/operation_derived1.csv")
operation <- fread("/home/luojiawei/inspire_benchmark_data/operation_derived1.csv")

source("data_preprocessing/4_sample_selection.R")
fwrite(operation_, "/home/luojiawei/inspire_benchmark_data/operation_.csv")
operation_ <- fread("/home/luojiawei/inspire_benchmark_data/operation_.csv")
dim(operation_)

root_path <- "/home/luojiawei/inspire_benchmark_data/all_op_id"
source("data_preprocessing/5_data_generation.R")



# files <- list.files(root_path)
# length(files)
# files[1]
# list.files(paste0(root_path, files[1]))
# read.csv(paste0(root_path, "/", files[1], "/", "lab.csv"))
# read.csv(paste0(root_path, "/", files[1], "/", "mask_lab.csv"))
# read.csv(paste0(root_path, "/", files[1], "/", "vit.csv"))
# read.csv(paste0(root_path, "/", files[1], "/", "mask_vit.csv"))
# read.csv(paste0(root_path, "/", files[1], "/", "ward_vit.csv"))
# read.csv(paste0(root_path, "/", files[1], "/", "mask_ward_vit.csv"))

# read.csv(paste0(root_path, "/", files[20], "/", "y_mat.csv"))[1:2,]
# read.csv(paste0(root_path, "/", files[20], "/", "y_static.csv"))

# > read.csv(paste0(root_path, "/", files[20], "/", "y_mat.csv"))[1:2,]
#   mbp hr
# 1   0  1

# > read.csv(paste0(root_path, "/", files[20], "/", "y_static.csv"))
#   death_30d have_icu have_aki have_ali postop_lung_complications
# 1         0        1        0        0                         0
#   postop_cardiac_complications postop_stroke
# 1                            0             0








