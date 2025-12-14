#!/bin/bash
# Combine multiple breast cancer datasets vertically with dataset ID column
# This is much faster than Python for large CSV files
# Output format: type (1st column), dataset_id (2nd column), then all other columns

OUTPUT_FILE="${1:-combined_breast_datasets.csv}"
TEMP_FILE="${OUTPUT_FILE}.tmp"

# Define input files
FILE1="/n/fs/vision-mix/jl0796/qcb/qcb455_project/Breast_GSE7904.csv"
FILE2="/n/fs/vision-mix/jl0796/qcb/qcb455_project/Breast_GSE26910.csv"
FILE3="/n/fs/vision-mix/jl0796/qcb/qcb455_project/Breast_GSE42568.csv"
FILE4="/n/fs/vision-mix/jl0796/qcb/qcb455_project/Breast_GSE45827.csv"

echo "Combining datasets to $OUTPUT_FILE..."
echo "Column order: type, dataset_id, samples, [gene columns...]"

# Step 1: Get the header from the first file and add dataset_id column
awk 'NR==1 {print $0",dataset_id"}' "$FILE1" > "$TEMP_FILE"

# Step 2: Process each file - skip header and add dataset_id
echo "Processing Breast_GSE7904.csv (dataset_id=1)..."
awk 'NR>1 {print $0",1"}' "$FILE1" >> "$TEMP_FILE"

echo "Processing Breast_GSE26910.csv (dataset_id=2)..."
awk 'NR>1 {print $0",2"}' "$FILE2" >> "$TEMP_FILE"

echo "Processing Breast_GSE42568.csv (dataset_id=3)..."
awk 'NR>1 {print $0",3"}' "$FILE3" >> "$TEMP_FILE"

echo "Processing Breast_GSE45827.csv (dataset_id=4)..."
awk 'NR>1 {print $0",4"}' "$FILE4" >> "$TEMP_FILE"

# Step 3: Reorder columns - move 'type' column to first position, dataset_id to second
# Assumes: samples (col 1), type (col 2), gene columns (col 3+), dataset_id (last col)
# Output: type, dataset_id, samples, gene columns
echo "Reordering columns (moving 'type' to first position)..."
awk -F',' 'BEGIN {OFS=","}
NR==1 {
    # Header: type, dataset_id, samples, rest of columns (excluding type and dataset_id)
    n = NF
    printf "type,dataset_id,samples"
    for (i=3; i<n; i++) printf ",%s", $i
    printf "\n"
}
NR>1 {
    # Data rows: reorder as type, dataset_id, samples, rest
    n = NF
    printf "%s,%s,%s", $2, $n, $1
    for (i=3; i<n; i++) printf ",%s", $i
    printf "\n"
}' "$TEMP_FILE" > "$OUTPUT_FILE"

# Clean up temp file
rm "$TEMP_FILE"

# Count lines in output
TOTAL_LINES=$(wc -l < "$OUTPUT_FILE")
echo ""
echo "Done! Combined dataset has $TOTAL_LINES lines (including header)"
echo "Output saved to: $OUTPUT_FILE"
