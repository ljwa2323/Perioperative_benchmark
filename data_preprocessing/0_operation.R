# This script processes the operations dataset for perioperative medicine research. 
# It calculates various metrics such as BMI, length of stay, operation duration, 
# and merges additional information regarding surgery sites based on ICD-10-PCS codes. 
# The final processed data is saved as a CSV file for further analysis.

# Read the operations dataset from a CSV file
print("reading operations.csv")
operation <- fread(file.path(inspire_path, "operations.csv"), header=T)
print("operations.csv has been read")

print("calculating BMI")
# Calculate BMI (Body Mass Index) for each operation
operation$bmi <- operation$weight / (operation$height/100)^2

print("calculating length of stay (LOS)")
# Calculate length of stay (LOS) by subtracting admission time from discharge time
operation$los <- operation$discharge_time - operation$admission_time

print("calculating operation duration")
# Calculate operation duration by subtracting operation start time from operation end time
operation$op_duration <- operation$opend_time - operation$opstart_time

print("calculating operating room duration")
# Calculate operating room duration by subtracting the time of entering the OR from the time of leaving the OR
operation$or_duration <- operation$orout_time - operation$orin_time

print("calculating anesthesia duration")
# Calculate anesthesia duration by subtracting anesthesia start time from anesthesia end time
operation$an_duration <- operation$anend_time - operation$anstart_time

print("calculating CPB duration")
# Calculate CPB (Cardiopulmonary Bypass) duration by subtracting CPB start time from CPB end time
operation$cpb_duration <- operation$cpboff_time - operation$cpbon_time

print("calculating ICU duration")
# Calculate ICU (Intensive Care Unit) duration by subtracting ICU admission time from ICU discharge time
operation$icu_duration <- operation$icuout_time - operation$icuin_time

print("converting sex to numeric")
operation$sex <- ifelse(operation$sex == "F", 0, 1)

print("replacing NA values in operation duration with 0")
operation$op_duration[is.na(operation$op_duration)] <- 0

print("replacing NA values in operating room duration with 0")
operation$or_duration[is.na(operation$or_duration)] <- 0

print("replacing NA values in anesthesia duration with 0")
operation$an_duration[is.na(operation$an_duration)] <- 0

print("replacing NA values in ICU duration with 0")
operation$icu_duration[is.na(operation$icu_duration)] <- 0
operation$have_icu <- ifelse(operation$icu_duration > 0, 1, 0)
print("replacing NA values in CPB duration with 0")
operation$cpb_duration[is.na(operation$cpb_duration)] <- 0
operation$have_cpb <- ifelse(operation$cpb_duration > 0, 1, 0)
print("creating binary variable for 30-day mortality")
operation$death_30d <- ifelse(is.na(operation$inhosp_death_time), 0, 
                               ifelse((operation$inhosp_death_time - operation$orout_time) <= 30 * 24 * 60, 1, 0))

NAME <- names(operation)

print("creating lookup table for surgery site based on PCS type")
pcs_type_lookup <- data.frame(
  pcs_type = c("08", "09", "0C", "0H", "0J", "0K", "0T", "0U", "0V", "10", "0D", "0F", "00", "0G", "0B", "0P", "0Q", "0N", "0S", "0R", "0L", "0M", "02", "03", "04", "05", "06", "07", "0W", "0Y", "0X", "01", "0E", "0A"),
  surgery_site = c("Head_and_Neck", "Head_and_Neck", "Head_and_Neck", "Skin_and_Soft_Tissue", "Skin_and_Soft_Tissue", "Skin_and_Soft_Tissue", "Urinary_and_Reproductive_Systems", "Urinary_and_Reproductive_Systems", "Urinary_and_Reproductive_Systems", "Urinary_and_Reproductive_Systems", "Gastrointestinal_System", "Hepatobiliary_System_and_Pancreas", "Central_Nervous_System_and_Cranial_Nerves", "Endocrine_System", "Respiratory_System", "Bones_and_Joints", "Bones_and_Joints", "Bones_and_Joints", "Bones_and_Joints", "Bones_and_Joints", "Bones_and_Joints", "Bones_and_Joints", "Heart_and_Great_Vessels", "Peripheral_Vascular_System", "Peripheral_Vascular_System", "Peripheral_Vascular_System", "Peripheral_Vascular_System", "Lymphatic_and_Hemic_Systems", "Other", "Other", "Other", "Other", "Other", "Other")
)

print("extracting the first two characters from the ICD-10-PCS code to create a new column for PCS type")
operation$pcs_type <- substr(operation$icd10_pcs, 1, 2)

print("merging the operation dataset with the lookup table to add surgery site information")
operation <- merge(operation, pcs_type_lookup, by = "pcs_type", all.x = TRUE)
operation <- as.data.frame(operation)
operation <- operation[, c(NAME, "surgery_site")]
