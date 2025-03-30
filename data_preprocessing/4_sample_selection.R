# View the number of unique op_id and subject_id in the original data
cat("Original data:\n")
cat("Number of unique op_id:", length(unique(operation$op_id)), "\n")
cat("Number of unique subject_id:", length(unique(operation$subject_id)), "\n")
cat("Total rows:", nrow(operation), "\n\n")

# Step 1: Filter for general anesthesia surgeries
operation_ <- operation[antype == "General", ]
cat("After filtering for general anesthesia surgeries:\n")
cat("Number of unique op_id:", length(unique(operation_$op_id)), "\n")
cat("Number of unique subject_id:", length(unique(operation_$subject_id)), "\n")
cat("Total rows:", nrow(operation_), "\n\n")

# Step 2: Filter for surgeries with age > 16
operation_ <- operation_[age > 16, ]
cat("After filtering for age > 16:\n")
cat("Number of unique op_id:", length(unique(operation_$op_id)), "\n")
cat("Number of unique subject_id:", length(unique(operation_$subject_id)), "\n")
cat("Total rows:", nrow(operation_), "\n\n")

# Step 2.5: Filter out surgeries with negative duration values
operation_ <- operation_[(!is.na(op_duration) & op_duration >= 0) & 
                         (is.na(or_duration) | or_duration >= 0) & 
                         (is.na(cpb_duration) | cpb_duration >= 0) & 
                         (is.na(an_duration) | an_duration >= 0) & 
                         (is.na(icu_duration) | icu_duration >= 0), ]
cat("After filtering out negative duration values:\n")
cat("Number of unique op_id:", length(unique(operation_$op_id)), "\n")
cat("Number of unique subject_id:", length(unique(operation_$subject_id)), "\n")
cat("Total rows:", nrow(operation_), "\n\n")

# Step 3: Filter for surgeries with duration >= 15 minutes
operation_ <- operation_[op_duration >= 15, ]
cat("After filtering for op_duration >= 15 minutes:\n")
cat("Number of unique op_id:", length(unique(operation_$op_id)), "\n")
cat("Number of unique subject_id:", length(unique(operation_$subject_id)), "\n")
cat("Total rows:", nrow(operation_), "\n\n")

# Step 4: Group by subject_id
# First, get unique subject_id
unique_subjects <- unique(operation_$subject_id)
n_subjects <- length(unique_subjects)

# Set a random seed to ensure reproducibility
set.seed(42)

# Randomly shuffle subject_id
shuffled_subjects <- sample(unique_subjects)

# Split according to the ratio of 7:2:1
train_size <- floor(n_subjects * 0.7)
test_size <- floor(n_subjects * 0.2)
val_size <- n_subjects - train_size - test_size

# Split subject_id
train_subjects <- shuffled_subjects[1:train_size]
test_subjects <- shuffled_subjects[(train_size+1):(train_size+test_size)]
val_subjects <- shuffled_subjects[(train_size+test_size+1):n_subjects]

# Create a mapping function to map subject_id to the corresponding dataset number
get_dataset <- function(subject_id) {
  if (subject_id %in% train_subjects) return(1)  # Training set
  if (subject_id %in% test_subjects) return(2)   # Test set
  if (subject_id %in% val_subjects) return(3)    # Validation set
  return(NA)  # This case should not occur
}

# Add dataset column for each surgery
operation_[, dataset := sapply(subject_id, get_dataset)]

# Check grouping results
cat("Grouping results:\n")
cat("Training set (1) count:", sum(operation_$dataset == 1), "rows, proportion:", 
    round(sum(operation_$dataset == 1) / nrow(operation_) * 100, 2), "%\n")
cat("Test set (2) count:", sum(operation_$dataset == 2), "rows, proportion:", 
    round(sum(operation_$dataset == 2) / nrow(operation_) * 100, 2), "%\n")
cat("Validation set (3) count:", sum(operation_$dataset == 3), "rows, proportion:", 
    round(sum(operation_$dataset == 3) / nrow(operation_) * 100, 2), "%\n\n")

# Check the number of unique subject_id in each dataset
cat("Number of unique subject_id in training set:", length(unique(operation_[dataset == 1, subject_id])), 
    ", proportion:", round(length(unique(operation_[dataset == 1, subject_id])) / n_subjects * 100, 2), "%\n")
cat("Number of unique subject_id in test set:", length(unique(operation_[dataset == 2, subject_id])), 
    ", proportion:", round(length(unique(operation_[dataset == 2, subject_id])) / n_subjects * 100, 2), "%\n")
cat("Number of unique subject_id in validation set:", length(unique(operation_[dataset == 3, subject_id])), 
    ", proportion:", round(length(unique(operation_[dataset == 3, subject_id])) / n_subjects * 100, 2), "%\n")

