# This script extracts pre-operative diagnoses for each operation from the diagnosis dataset.
# It reads the diagnosis data, processes each operation to identify pre-operative diagnoses,
# and merges the results back into the main dataset. The script handles different types of diagnoses
# and ensures that the results are properly formatted and merged for further analysis.


print("start to extract pre-operative diagnoses")

operation <- as.data.frame(operation)  # Convert to data frame for easier manipulation

# Read the diagnosis data from the CSV file
diag <- fread(file.path(inspire_path, "diagnosis.csv"), header=T)
diag <- as.data.frame(diag)  # Convert to data frame for easier manipulation

# Convert chart_time from minutes to hours
diag$chart_time <- diag$chart_time / 60

# Convert orin_time to numeric for comparison
operation$orin_time <- as.numeric(as.character(operation$orin_time))

print("extracting the pre-operative diagnosis for each operation")

diag <- diag[order(diag$subject_id), ]
subject_indices <- split(1:nrow(diag), diag$subject_id)

result_list <- pblapply(1:nrow(operation), function(i) {
    current_op_id <- operation$op_id[i]
    current_subject_id <- operation$subject_id[i]
    current_orin_time <- operation$orin_time[i]
    
    subject_rows <- subject_indices[[as.character(current_subject_id)]]
    
    if (!is.null(subject_rows)) {
        
        preop_rows <- subject_rows[diag$chart_time[subject_rows] < current_orin_time]
        
        if (length(preop_rows) > 0) {
            
            preop_diags <- unique(diag[preop_rows, c("subject_id", "icd10_cm"), drop=F])
            preop_diags$op_id <- current_op_id
            return(preop_diags)
        }
    }
    
    return(NULL)
})

result_list <- result_list[!sapply(result_list, is.null)]

# Combine all pre-operative diagnoses into a single data frame
preop_diags <- do.call(rbind, result_list)

# Print message indicating the end of extraction
print("end of extracting the pre-operative diagnosis for each operation")

# Define a list of diseases with their corresponding ICD-10 codes
diseases <- list(
  preop_essential_hypertension = "I10",  # Essential Hypertension
  preop_coronary_heart_disease = paste0("I",str_pad(20:25, 2, "left", "0")),  # Coronary Heart Disease
  preop_congestive_heart_failure = "I50",  # Congestive Heart Failure
  preop_atrial_fibrillation_and_flutter = "I48",  # Atrial Fibrillation and Flutter
  preop_abnormalities_of_heart_beat = "R00",  # Abnormalities of Heart Beat
  preop_diabetes_mellitus = paste0("E",str_pad(10:14, 2, "left", "0")),  # Diabetes Mellitus
  preop_cerebral_infarction = "I63",  # Cerebral Infarction
  preop_transient_cerebral_ischemic_attacks = "G45",  # Transient Cerebral Ischemic Attacks
  preop_emphysema_or_copd = paste0("J",43:44),  # Emphysema or COPD
  preop_asthma = "J45",  # Asthma
  preop_acute_upper_respiratory_infections = paste0("J",str_pad(0:6, 2, "left", "0")),  # Acute Upper Respiratory Infections
  preop_acute_lower_respiratory_infections = paste0("J",str_pad(9:22, 2, "left", "0")),  # Acute Lower Respiratory Infections
  preop_abnormalities_of_breathing = "R06",  # Abnormalities of Breathing
  preop_malignant_neoplasms = paste0("C",str_pad(0:97, 2, "left", "0")),  # Malignant Neoplasms
  preop_in_situ_neoplasms = paste0("D",str_pad(0:9, 2, "left", "0")),  # In Situ Neoplasms
  preop_benign_neoplasms = paste0("D",str_pad(10:36, 2, "left", "0")),  # Benign Neoplasms
  preop_neoplasms_of_uncertain_behavior = paste0("D",str_pad(37:48, 2, "left", "0")),  # Neoplasms of Uncertain Behavior
  preop_chronic_kidney_disease = "N18",  # Chronic Kidney Disease
  preop_chronic_viral_hepatitis = "B18",  # Chronic Viral Hepatitis
  preop_liver_disease = paste0("K",str_pad(70:77, 2, "left", "0")),  # Liver Disease
  preop_gastro_esophageal_reflux_disease = "K21",  # Gastro Esophageal Reflux Disease
  preop_anemia = paste0("D",str_pad(50:64, 2, "left", "0")),  # Anemia
  preop_disorder_of_thyroid = paste0("E",str_pad(0:7, 2, "left", "0"))  # Disorder of Thyroid
)

diseases_regex <- lapply(diseases, function(code) {
  if (length(code) == 1) {
    paste0("^", code) 
  } else {
    paste0("^(", paste(code, collapse="|"), ")") 
  }
})

# Initialize each disease column in the operation table to 0
for (disease in names(diseases)) {
  operation[[disease]] <- 0
}

print("start to check each operation for the presence of diseases based on pre-operative diagnoses")


setDT(preop_diags)

operation[, names(diseases)] <- t(pbsapply(1:nrow(operation), function(i) {
  current_diags <- preop_diags[preop_diags$op_id == operation$op_id[i], icd10_cm] 
  
  sapply(diseases_regex, function(regex) {
    as.numeric(any(grepl(regex, current_diags))) 
  })
}))


print("pre-operative diagnoses have been checked")

print("start to extract pre-operative vitals")

ward_vitals <- fread(file.path(inspire_path, "ward_vitals.csv"),header=T)
ward_vitals_ <- ward_vitals[which(ward_vitals$item_name %in% c("crrt","ecmo", "vent")),]


ward_vitals_ <- ward_vitals_[order(ward_vitals_$subject_id), ]


subject_indices <- split(1:nrow(ward_vitals_), ward_vitals_$subject_id)


process_preop_vitals <- function(operation_row) {
    
    subject_rows <- subject_indices[[as.character(operation_row$subject_id)]]
    
    if (is.null(subject_rows)) {
        subject_vit <- ward_vitals_[0, ]
    } else {
        subject_vit <- ward_vitals_[subject_rows[ward_vitals_$chart_time[subject_rows] <= operation_row$orin_time & 
                                              ward_vitals_$chart_time[subject_rows] >= operation_row$admission_time], ]
    }
    if (nrow(subject_vit) == 0) {
        preop_crrt <- 0
        preop_ecmo <- 0
        preop_vent <- 0
    } else {
        preop_crrt <- ifelse(any(subject_vit$item_name == "crrt"), 1, 0)
        preop_ecmo <- ifelse(any(subject_vit$item_name == "ecmo"), 1, 0)
        preop_vent <- ifelse(any(subject_vit$item_name == "vent"), 1, 0)
    }
    
    return(data.frame(
        op_id = operation_row$op_id,
        subject_id = operation_row$subject_id,
        preop_crrt = preop_crrt,
        preop_ecmo = preop_ecmo,
        preop_vent = preop_vent
    ))
}

results_list <- pblapply(1:nrow(operation), function(i) {
    process_preop_vitals(operation[i,])
})


preop_vitals <- do.call(rbind, results_list)

operation <- merge(operation, preop_vitals, by=c("op_id","subject_id"), all.x=T)

print("pre-operative vitals have been extracted")