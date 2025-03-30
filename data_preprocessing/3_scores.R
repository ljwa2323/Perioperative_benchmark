# This script calculates various scores for operations data, including SORT, CCI, and RCRI scores.
# It processes the operations dataset and diagnosis data to compute these scores, which are then
# used for further analysis. The script also handles missing values and merges additional information
# regarding surgery sites based on ICD-10-PCS codes.

# The scores calculated in this script are:
# 1. SORT (Surgical Outcome Risk Tool) score
# 2. CCI (Charlson Comorbidity Index) score
# 3. RCRI (Revised Cardiac Risk Index) score

# The script uses data.table for efficient data manipulation and includes functions to check ICD-10 codes
# and calculate the scores based on predefined criteria.


print("Starting to calculate SORT score")

setDT(operation)
setDT(preop_diags)

# Sort preop_diags by op_id
setkey(preop_diags, op_id)

# Create a function to check if the ICD10 code is within the specified range
check_icd10 <- function(icd10_code) {
  # Check if it matches C00-D49 or Z85
  if (str_detect(icd10_code, "^C[0-9]+|^D[0-4][0-9]") | str_detect(icd10_code, "^Z85")) {
    return(TRUE)
  } else {
    return(FALSE)
  }
}

# Add a new column to operation, initialized to 0
operation[, icd10_match := 0]

# Get all unique op_id
unique_op_ids <- unique(operation$op_id)

# Use pblapply instead of lapply to show progress bar
results <- pblapply(unique_op_ids, function(current_op_id) {
  # Get all icd10_cm codes corresponding to the op_id
  subject_icd10_codes <- preop_diags[op_id == current_op_id, icd10_cm]
  
  # Check if any codes match C00-D49 or Z85
  if (length(subject_icd10_codes) > 0 && any(sapply(subject_icd10_codes, check_icd10))) {
    return(1)
  } else {
    return(0)
  }
})

# Convert results to a data table and update operation
result_dt <- data.table(op_id = unique_op_ids, icd10_match = unlist(results))
operation[result_dt, icd10_match := i.icd10_match, on = "op_id"]



##  ===============================  SORT Score ===============================

# Recalculate SORT score using data.table
operation[, sort_score := {
  # 1. ASA score
  asa_score <- fifelse(asa == 3, 1.411, fifelse(asa == 4, 2.388, fifelse(asa == 5, 4.081, 0)))
  
  # 2. EMOP score
  emop_score <- fifelse(emop == 1, 1.657, 0)
  
  # 3. Surgery type score
  surgery_type_score <- fifelse(substr(icd10_pcs, 1, 2) %in% c("0D", "0F", "0G", "0W", "0X", "0Y", "02"), 0.712, 0)
  
  # 4. Surgery duration score
  surgery_duration_score <- fifelse(op_duration > 120, 0.381, 0)
  
  # 5. ICD10 match score
  icd10_match_score <- fifelse(icd10_match == 1, 0.667, 0)
  
  # 6. Age score
  age_score <- fifelse(age >= 65 & age <= 79, 0.777, fifelse(age >= 80, 1.591, 0))
  
  # Total score
  asa_score + emop_score + surgery_type_score + surgery_duration_score + icd10_match_score + age_score
}, by = .(op_id)]

print("SORT score calculation completed")

#  ==============================  CCI Score ===========================

print("Starting to calculate CCI score")

# Disease codes with 1 point
score_1_codes <- c("I21", "I50", "I73", "F01", "I63", "M35", "K25", "K26", "K27", "E11", "J44", "K70")
# Disease codes with 2 points
score_2_codes <- c("I69", "N18")
score_2_codes <- c(score_2_codes, 
                    paste0("C",str_pad(0:99, 2, "left", "0")),
                    paste0("D",str_pad(0:49, 2, "left", "0")))
# Disease codes with 3 points
score_3_codes <- c("K74", "K75", "K76")
# Disease codes with 6 points
score_6_codes <- c("C77", "C78", "C79", "C80", "B20", "B21", "B22", "B23", "B24")

preop_diags$score <- rep(0, nrow(preop_diags))
preop_diags$score[preop_diags$icd10_cm %in% score_1_codes] <- 1
preop_diags$score[preop_diags$icd10_cm %in% score_2_codes] <- 2
preop_diags$score[preop_diags$icd10_cm %in% score_3_codes] <- 3
preop_diags$score[preop_diags$icd10_cm %in% score_6_codes] <- 6

# Ensure preop_diags and operation are of data.table type
setDT(preop_diags)
setDT(operation)

# Iterate over each row in operation
operation[, CCI_score := {
  # Get the current row's op_id
  current_op_id <- op_id
  
  # Get all diagnosis scores corresponding to the op_id
  subject_scores <- preop_diags[op_id == current_op_id, .(icd10_cm, score)]
  
  # Keep only one for the same icd10_cm code
  unique_subject_scores <- unique(subject_scores, by = "icd10_cm")
  
  # Sum the selected scores
  sum(unique_subject_scores$score)
}, by = .(op_id)]

print("CCI score calculation completed")


# =============================== RCRI Score ==============================

print("Starting to calculate RCRI score")

# Ischemic heart disease history I20-I25 1 point
# Congestive heart failure history I50 1 point
# Stroke history / Transient ischemic attack history / Intracranial hemorrhage history I60-I63 or G45 1 point
# Diabetes management E11 1 point
# Creatinine labs (creatinine > 2 mg/dL) 1 point
# Surgical risk assessment - surgical site icd10_pcs (Abdominal surgery: 0D, 0F, 0G, 0T, 0U, 0V, 10, Thoracic surgery: 0B, Cranial surgery: 00, Major vascular surgery: 02, Upper vascular surgery: 03, 05) 1 point

# Define ICD-10 codes for various diseases
ischemic_heart_codes <- c("I20", "I21", "I22", "I23", "I24", "I25")  # Ischemic heart disease
heart_failure_codes <- "I50"  # Congestive heart failure
stroke_codes <- c("I60", "I61", "I62", "I63", "G45")  # Stroke/TIA/intracranial hemorrhage
diabetes_codes <- "E11"  # Diabetes

# Combine all relevant disease codes
relevant_icd10_codes <- c(ischemic_heart_codes, heart_failure_codes, stroke_codes, diabetes_codes)
diag_relevant <- preop_diags[icd10_cm %in% relevant_icd10_codes]

# Filter records with creatinine greater than 2 mg/dL
labs_relevant <- labs[item_name == "creatinine" & value > 2]
# Sort by subject_id
setkey(labs_relevant, subject_id)
# Create an index for subject_id
subject_ids <- unique(labs_relevant$subject_id)
subject_index <- list()
for (id in subject_ids) {
  rows <- which(labs_relevant$subject_id == id)
  subject_index[[as.character(id)]] <- c(min(rows), max(rows))
}

# Surgical site corresponding icd10_pcs codes
surgery_icd10_pcs_codes <- c("0D", "0F", "0G", "0T", "0U", "0V", "10", "0B", "00", "02", "03", "05")

# Iterate over each row in operation to calculate total score
operation[, RCRI_score := {
  current_subject_id <- subject_id
  current_op_id <- op_id
  current_orin_time <- orin_time
  current_admission_time <- admission_time
  
  # Get all disease codes corresponding to the current op_id
  subject_diags <- diag_relevant[op_id == current_op_id, .(icd10_cm)]
  
  # Calculate scores for each type of disease
  ischemic_heart_score <- as.integer(any(subject_diags$icd10_cm %in% ischemic_heart_codes))
  heart_failure_score <- as.integer(any(subject_diags$icd10_cm %in% heart_failure_codes))
  stroke_score <- as.integer(any(subject_diags$icd10_cm %in% stroke_codes))
  diabetes_score <- as.integer(any(subject_diags$icd10_cm %in% diabetes_codes))
  
  # Calculate creatinine score
  creatinine_score <- 0
  if (as.character(current_subject_id) %in% names(subject_index)) {
    idx <- subject_index[[as.character(current_subject_id)]]
    start_row <- idx[1]
    end_row <- idx[2]
    
    # Use the index range to filter by time
    relevant_labs <- labs_relevant[start_row:end_row]
    creatinine_score <- as.integer(sum(relevant_labs[chart_time >= current_admission_time & 
                                                    chart_time <= current_orin_time, .N] > 0))
  }

  # Calculate surgical site score
  surgery_score <- ifelse(substr(icd10_pcs, 1, 2) %in% surgery_icd10_pcs_codes, 1, 0)
  
  # Total score (maximum 6 points)
  sum(c(ischemic_heart_score, heart_failure_score, stroke_score, diabetes_score, 
        creatinine_score, surgery_score), na.rm=TRUE)
}, by = .(op_id)]

print("RCRI score calculation completed")

operation[, icd10_match := NULL]


