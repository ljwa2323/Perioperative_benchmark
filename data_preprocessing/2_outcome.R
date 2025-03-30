# This script processes the outcome data for operations, including the calculation of various postoperative complications.
# It reads the necessary data, processes each operation to identify complications, and merges the results back into the main dataset.
# The script handles different types of complications, such as postoperative unplanned intubation and cardiac performance index,
# and ensures that the results are properly formatted and merged for further analysis.

# The main steps in this script include:
# 1. Reading and preparing the necessary data.
# 2. Defining functions to process each type of complication.
# 3. Applying these functions to the operations dataset.
# 4. Merging the results back into the main dataset.
# 5. Printing messages to indicate the progress and completion of each step.

print("start calculating postop_cardiac_performance_index")

labs <- fread(file.path(inspire_path, "labs.csv"), header=T)
# Filter and process troponin_t data
labs_ <- labs[which(labs$item_name %in% c("troponin_t")),]
labs_$value <- labs_$value * 1000

# Sort by subject_id
labs_ <- labs_[order(labs_$subject_id), ]

# Create subject_id index
subject_indices <- split(1:nrow(labs_), labs_$subject_id)

process_operation <- function(operation_row) {
    # Get the data range for the patient
    subject_rows <- subject_indices[[as.character(operation_row$subject_id)]]
    
    if (!is.null(subject_rows)) {
        # Filter time within the patient's data range
        pre_rows <- subject_rows[labs_$chart_time[subject_rows] <= operation_row$orin_time & 
                               labs_$chart_time[subject_rows] >= operation_row$admission_time]
        post_rows <- subject_rows[labs_$chart_time[subject_rows] >= operation_row$orout_time & 
                                labs_$chart_time[subject_rows] <= operation_row$discharge_time]
        
        # Get the first and last values
        v1 <- if(length(pre_rows) > 0) get_first(labs_$value[pre_rows]) else NA
        v2 <- if(length(post_rows) > 0) get_last(labs_$value[post_rows]) else NA
    } else {
        v1 <- NA
        v2 <- NA
    }
    
    # Initialize the flag as NA
    C_PMI <- NA
    C_MINS <- NA
    
    # Check if v1 and v2 are not NA
    if (!is.na(v1) && !is.na(v2)) {
        delta <- v2 - v1
        C_PMI <- as.integer(delta > 14)
        C_MINS <- as.integer((v2 >= 20 && delta >= 5) || v2 >= 65)
    } else if (!is.na(v2)) {
        C_MINS <- as.integer(v2 >= 65)
    }
    
    return(data.frame(op_id = operation_row$op_id, postop_cardiac_performance_index = C_PMI, postop_cardiac_function_index = C_MINS))
}

# Use lapply to process all operation records
total_operations <- nrow(operation)
results_list <- pblapply(1:total_operations, function(i) {
    process_operation(operation[i,])
}, cl = NULL)
cat("\nCardiac performance index calculation completed!\n")

heart_injury <- do.call(rbind, results_list)

operation <- merge(operation, heart_injury, by='op_id', all.x=T)
rm("heart_injury")

print("postop_cardiac_performance_index calculated...")

operation <- as.data.frame(operation)

print("start calculating postoperative continuous renal replacement therapy, extracorporeal membrane oxygenation, ventilation...")

ward_vitals_ <- ward_vitals[which(ward_vitals$item_name %in% c("crrt","ecmo", "vent")),]

ward_vitals_ <- ward_vitals_[order(ward_vitals_$subject_id), ]

subject_indices <- split(1:nrow(ward_vitals_), ward_vitals_$subject_id)

process_postop_vitals <- function(operation_row) {
    
    subject_rows <- subject_indices[[as.character(operation_row$subject_id)]]
    
    if (is.null(subject_rows)) {
        # If no corresponding records are found
        subject_vit <- ward_vitals_[0, ]
    } else {
        # Filter time only within the patient's data range
        subject_vit <- ward_vitals_[subject_rows[ward_vitals_$chart_time[subject_rows] >= operation_row$orout_time & 
                                              ward_vitals_$chart_time[subject_rows] <= operation_row$discharge_time], ]
    }
    
    if (nrow(subject_vit) == 0) {
        postop_crrt <- 0
        postop_ecmo <- 0
        postop_vent <- 0
    } else {
        postop_crrt <- ifelse(any(subject_vit$item_name == "crrt"), 1, 0)
        postop_ecmo <- ifelse(any(subject_vit$item_name == "ecmo"), 1, 0)
        postop_vent <- ifelse(any(subject_vit$item_name == "vent"), 1, 0)
    }
    
    return(data.frame(
        op_id = operation_row$op_id,
        subject_id = operation_row$subject_id,
        postop_continuous_renal_replacement_therapy = postop_crrt,
        postop_extracorporeal_membrane_oxygenation = postop_ecmo,
        postop_ventilation = postop_vent
    ))
}
results_list <- pblapply(1:nrow(operation), function(i) {
  process_postop_vitals(operation[i,])
})

# Combine results into a data frame
postop_vitals <- do.call(rbind, results_list)

operation <- merge(operation, postop_vitals, by=c("op_id","subject_id"), all.x=T)
rm("postop_vitals")

print("postop_continuous_renal_replacement_therapy, extracorporeal_membrane_oxygenation, ventilation calculated...")

operation[1:2,]

print("start calculating postop_acute_kidney_injury...")

labs_ <- labs[which(labs$item_name %in% c("creatinine")),]
labs_ <- labs_[!duplicated(labs_[,c("subject_id","chart_time")], ), ]
labs_ <- labs_[order(labs_$subject_id, labs_$chart_time, decreasing = F), ]
subject_indices <- split(1:nrow(labs_), labs_$subject_id)


process_aki <- function(operation_row) {
    
    subject_rows <- subject_indices[[as.character(operation_row$subject_id)]]
    
    if (!is.null(subject_rows)) {
        
        pre_rows <- subject_rows[labs_$chart_time[subject_rows] <= operation_row$orin_time]
        post_rows <- subject_rows[labs_$chart_time[subject_rows] >= operation_row$orout_time]
        
       
        if (length(pre_rows) > 0 && length(post_rows) > 0) {
            t1 <- get_last(labs_$chart_time[pre_rows])
            v1 <- get_last(labs_$value[pre_rows])
            t2 <- get_first(labs_$chart_time[post_rows])
            v2 <- get_first(labs_$value[post_rows])
        } else {
            t1 <- t2 <- v1 <- v2 <- NA
        }
    } else {
        t1 <- t2 <- v1 <- v2 <- NA
    }
    
    
    AKI <- NA
    crrt <- operation_row$postop_continuous_renal_replacement_therapy
    
    if (!is.na(v1) && !is.na(v2)) {
        ratio <- v2 / v1
        increase <- v2 - v1
        time_diff_hours <- (t2 - t1) / 60  
        
        # Calculate AKI level based on rules
        if (!is.na(crrt) && (v2 >= 4 || crrt == 1)) {
            AKI <- 3
        } else if ((ratio >= 3) && time_diff_hours <= 168) {  
            AKI <- 3
        } else if (ratio >= 2 && ratio < 3 && time_diff_hours <= 168) {  
            AKI <- 2
        } else if ((ratio >= 1.5 && ratio < 2 && time_diff_hours <= 168) || (increase >= 0.3 && time_diff_hours <= 48)) {
            AKI <- 1
        } else {
            AKI <- 0
        }
    }
    
    return(data.frame(op_id = operation_row$op_id, postop_acute_kidney_injury = AKI))
}


# Use lapply to process all operation records
total_operations <- nrow(operation)
results_list <- pblapply(1:total_operations, function(i) {
    process_aki(operation[i,])
})

cat("\nAcute Kidney Injury (AKI) calculation completed!\n")

acute_kidney_injury <- do.call(rbind, results_list)

operation <- merge(operation, acute_kidney_injury, by='op_id', all.x=T)
operation$have_aki <- ifelse(operation$postop_acute_kidney_injury > 0, 1, 0)
rm("acute_kidney_injury")

print("postop_acute_kidney_injury calculated...")



print("start calculating postop_acute_liver_injury...")

labs_ <- labs[which(labs$item_name %in% c("alt")),]
labs_ <- labs_[order(labs_$subject_id), ]


subject_indices <- split(1:nrow(labs_), labs_$subject_id)


process_ali <- function(operation_row) {

    subject_rows <- subject_indices[[as.character(operation_row$subject_id)]]
    
    if (!is.null(subject_rows)) {

        pre_rows <- subject_rows[labs_$chart_time[subject_rows] <= operation_row$opstart_time]
        post_rows <- subject_rows[labs_$chart_time[subject_rows] >= operation_row$opend_time]
        

        v1 <- if(length(pre_rows) > 0) get_last(labs_$value[pre_rows]) else NA
        v2 <- if(length(post_rows) > 0) get_first(labs_$value[post_rows]) else NA
    } else {
        v1 <- v2 <- NA
    }
    
    # Initialize ALI level as NA
    ALI <- NA
    death <- operation_row$death_30d
    
    # Define normal upper limit based on gender
    normal_upper_limit <- ifelse(operation_row$sex == 0, 35, 40)  # 0 for female, 1 for male
    
    # Check if v1 and v2 are both not NA
    if (!is.na(v1) && !is.na(v2)) {
        ratio <- v2 / normal_upper_limit
        
        # Calculate ALI level based on rules
        if (death == 1) {
            ALI <- 5
        } else if (ratio > 20) {
            ALI <- 4
        } else if (ratio > 5 && ratio <= 20) {
            ALI <- 4
        } else if (ratio > 3 && ratio <= 5) {
            ALI <- 2
        } else if (ratio > 2 && ratio <= 3) {
            ALI <- 1
        } else {
            ALI <- 0
        }
    }
    
    return(data.frame(op_id = operation_row$op_id, postop_acute_liver_injury = ALI))
}

# Use lapply to process all surgical records
total_operations <- nrow(operation)
results_list <- pblapply(1:total_operations, function(i) {
    process_ali(operation[i,])
})
cat("\nAcute Liver Injury (ALI) indicator calculation completed!\n")

# Combine results into a data frame
liver_injury <- do.call(rbind, results_list)

table(liver_injury$ALI)

operation <- merge(operation, liver_injury, by='op_id', all.x=T)
rm("liver_injury")

print("postop_acute_liver_injury calculated...")
operation <- as.data.frame(operation)

operation$have_ali <- ifelse(operation$postop_acute_liver_injury > 0, 1, 0)
print("start calculating postop_postoperative_complication based on ICD code...")

# Define the ICD-10 code ranges for various complications
icd_comp_map <- list(
    postop_postoperative_complication = c("T81"),  # Postoperative complications
    postop_nervous_system_complication = c("G97"),  # Nervous system complications
    postop_digestive_system_complication = c("K91"),  # Digestive system complications
    postop_circulatory_system_complication = c("I97"),  # Circulatory system complications
    postop_musculoskeletal_system_complication = c("M96"),  # Musculoskeletal system complications
    postop_genitourinary_system_complication = c("N99"),  # Genitourinary system complications
    postop_skin_complication = c("L76"),  # Skin complications
    postop_eye_complication = c("H59"),  # Eye complications
    postop_ear_complication = c("H95"),  # Ear complications
    postop_endocrine_complication = c("E89"),  # Endocrine complications
    postop_hypotension = c("I95"),  # Hypotension
    postop_subarachnoid_hemorrhage = c("I60"),  # Subarachnoid hemorrhage
    postop_intracerebral_hemorrhage = c("I61", "I62"),  # Intracerebral hemorrhage
    postop_cerebral_infarction = c("I63"),  # Cerebral infarction
    postop_angina_pectoris = c("I20"),  # Angina pectoris
    postop_acute_myocardial_infarction = c("I21"),  # Acute myocardial infarction
    postop_acute_ischemic_heart_disease = c("I24"),  # Acute ischemic heart disease
    postop_chronic_ischemic_heart_disease = c("I25"),  # Chronic ischemic heart disease
    postop_atrial_fibrillation = c("I48"),  # Atrial fibrillation
    postop_cardiac_arrest = c("I46"),  # Cardiac arrest
    postop_pacemaker_trouble = c("Z45.0"),  # Pacemaker trouble
    postop_arrhythmia = c("I44", "I47"),  # Arrhythmia
    postop_pulmonary_embolism = c("I26"),  # Pulmonary embolism
    postop_venous_thrombosis = c("I80"),  # Venous thrombosis
    postop_portal_vein_thrombosis = c("I81"),  # Portal vein thrombosis
    postop_thromboembolism = c("I63", "I21", "I26", "I80", "I81", "I82"),  # Thromboembolism
    postop_fever = c("R50"),  # Fever
    postop_pain = c("R52"),  # Pain
    postop_postoperative_nausea_and_vomiting = c("R11"),  # Postoperative nausea and vomiting
    # Add pulmonary complications
    postop_atelectasis = c("J98"),  # Atelectasis
    postop_pulmonary_edema = c("J81"),  # Pulmonary edema
    postop_pneumothorax = c("J93"),  # Pneumothorax
    postop_pleural_effusion = c("J90", "J91"),  # Pleural effusion
    postop_pulmonary_infection = c("J12", "J13", "J14", "J15", "J16", "J17", "J18"),  # Pulmonary infection
    postop_respiratory_failure = c("J96"),  # Respiratory failure
    postop_respiratory_distress_syndrome = c("J80"),  # Respiratory distress syndrome
    postop_j95 = c("J95")  # Respiratory system complications
)


# 定义一个函数，用于检查特定的ICD-10编码是否属于某个并发症类别
check_complication <- function(icd_code) {

  # 检查ICD编码是否在任何并发症类别中
  for (complication in names(icd_comp_map)) {
    if (icd_code %in% icd_comp_map[[complication]]) {
      return(complication)
    }
  }
  return(NA)
}

# Assume diag has been sorted and indexed
diag <- diag[order(diag$subject_id), ]
subject_indices <- split(1:nrow(diag), diag$subject_id)

process_complications <- function(operation_row) {
    
    subject_rows <- subject_indices[[as.character(operation_row$subject_id)]]
    
    
    result <- setNames(
        rep(0, length(names(icd_comp_map))), 
        names(icd_comp_map)
    )
    
    if (!is.null(subject_rows)) {
        
        valid_rows <- subject_rows[
            diag$chart_time[subject_rows] > operation_row$orin_time & 
            diag$chart_time[subject_rows] < operation_row$discharge_time
        ]
        
        if (length(valid_rows) > 0) {
            # Check complications for each diagnosis
            for (icd_code in diag$icd10_cm[valid_rows]) {
                complication <- check_complication(icd_code)
                if (!is.na(complication)) {
                    result[complication] <- 1
                }
            }
        }
    }
    
    # Return data frame directly
    return(data.frame(
        op_id = operation_row$op_id,
        as.list(result),
        stringsAsFactors = FALSE
    ))
}

# Use lapply to process all surgical records and directly merge results
total_operations <- nrow(operation)
postop_comp_df <- do.call(rbind, pblapply(1:total_operations, function(i) {
    process_complications(operation[i,])
}))
cat("\nPostoperative complications based on ICD codes have been calculated!\n")

# Ensure all complication columns are numeric
postop_comp_df[,-1] <- lapply(postop_comp_df[,-1], as.numeric)

operation <- merge(operation, postop_comp_df, by='op_id', all.x=T)
rm("postop_comp_df")

print("Postoperative complications based on ICD code calculated...")


print("Start calculating postoperative unplanned intubation...")
# Prepare data: filter for vent-related records
ward_vitals_vent <- ward_vitals[which(ward_vitals$item_name == "vent"), ]
ward_vitals_vent <- ward_vitals_vent[order(ward_vitals_vent$subject_id), ]

# Create subject_id index
subject_indices <- split(1:nrow(ward_vitals_vent), ward_vitals_vent$subject_id)

# Define function to process a single surgical record
process_unplanned_intubation <- function(operation_row) {
    # Get the data range for the patient
    subject_rows <- subject_indices[[as.character(operation_row$subject_id)]]
    
    # Initialize result
    unplanned_intubation <- 0
    first_intubation_time <- NA
    
    if (!is.null(subject_rows)) {
        # Filter time within the patient's data range
        valid_rows <- subject_rows[
            ward_vitals_vent$chart_time[subject_rows] > operation_row$orout_time & 
            ward_vitals_vent$chart_time[subject_rows] < operation_row$discharge_time
        ]
        
        if (length(valid_rows) > 1) {  # At least two records are needed to detect changes
            # Get sorted records
            vent_records <- ward_vitals_vent[valid_rows, ]
            vent_records <- vent_records[order(vent_records$chart_time), ]
            
            # Check for value changes
            vent_changes <- diff(vent_records$value)
            intubation_idx <- which(vent_changes == 1)
            
            if (length(intubation_idx) > 0) {
                unplanned_intubation <- 1
                # Record the time of the first intubation
                first_intubation_time <- vent_records$chart_time[intubation_idx[1] + 1]
            }
        }
    }
    
    return(data.frame(
        op_id = operation_row$op_id,
        postop_unplanned_intubation = unplanned_intubation,
        postop_first_intubation_time = first_intubation_time
    ))
}

# Use lapply to process all surgical records
total_operations <- nrow(operation)
results_list <- pblapply(1:total_operations, function(i) {
    process_unplanned_intubation(operation[i,])
})
cat("\nCalculation of unplanned intubation has been completed!\n")

# Combine results into a data frame
unplanned_intubation_df <- do.call(rbind, results_list)

# Merge into the main data frame
operation <- merge(operation, unplanned_intubation_df[,c("op_id","postop_unplanned_intubation")], by='op_id', all.x=T)
rm("unplanned_intubation_df")

print("The calculation of postoperative unplanned intubation has been completed...")

print("Summary of postoperative lung complications...")

operation$postop_lung_complications <- ifelse(apply(operation[,c("postop_atelectasis",
"postop_pulmonary_edema","postop_pneumothorax","postop_pleural_effusion",
"postop_pulmonary_infection","postop_respiratory_failure","postop_respiratory_distress_syndrome",
"postop_unplanned_intubation","postop_j95")], 1, sum) == 0, 0, 1)

print("Summary of postoperative lung complications calculation completed...")



print("Start calculating postoperative cardiac complications...")

operation <- as.data.frame(operation)

# Cardiac complications
index <- c("postop_angina_pectoris","postop_acute_myocardial_infarction","postop_arrhythmia",
"postop_chronic_ischemic_heart_disease","postop_hypotension","postop_cardiac_arrest",
"postop_atrial_fibrillation","postop_cardiac_performance_index","postop_cardiac_function_index")

operation$postop_cardiac_complications <- ifelse(apply(operation[,index], 1, function(x) {
    sum(x, na.rm = T)
}) > 0, 1, 0)

# Stroke
index <- c("postop_cerebral_infarction","postop_intracerebral_hemorrhage",
"postop_subarachnoid_hemorrhage")

operation$postop_stroke <- ifelse(apply(operation[,index], 1, function(x) {
    sum(x, na.rm = T)
}) > 0, 1, 0)

# MACEs
operation$postop_maces <- ifelse(operation$postop_cardiac_complications == 1 | operation$postop_stroke == 1, 1, 0)

print("Postoperative cardiac complications calculation completed...")

print("All postoperative complications have been annotated...")

