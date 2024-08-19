import pandas as pd
from roc_calculator import ROCCalculator

'''# Replace 'your_file.xlsx' with the path to your Excel file
file_path = '/Volumes/pelvis/projects/tiago/iqa/marksheet/the_marksheet.xlsx'

# Read the Excel sheet into a DataFrame
df = pd.read_excel(file_path)

# Extract the 3rd, 4th, and 5th columns (columns are 0-indexed, so columns 2, 3, and 4)
selected_columns = df.iloc[:, [2, 3, 4]]

cleaned_df = selected_columns.dropna()'''

import json

# Path to your JSON file
json_file_path = '/Volumes/pelvis/projects/tiago/iqa/marksheet/the_json.json'

# Load the JSON data
with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)

# Extract the dictionaries
case_target = data['case_target']
case_pred = data['case_pred']
image_quality = data['image_quality']

# Create a DataFrame
df = pd.DataFrame(list(case_target.items()), columns=['Subject', 'Target'])
df['Prediction'] = df['Subject'].map(case_pred)
df['Image_Quality'] = df['Subject'].map(image_quality)

# Order the DataFrame by Image Quality
df = df.sort_values(by='Image_Quality', ascending=False)

# Boolean variable to determine stacking
stack = True  # Set to False for evenly distributed groups (20%,20%,20%,20%,20%) instead of (20%,40%,60%,80%,100%) for num_groups = 5

# Number of groups
num_groups = 5

# Create the groups dynamically
subject_groups = []
if stack:
    for i in range(num_groups):
        end_idx = len(df) * (i + 1) // num_groups
        group = df['Subject'].iloc[:end_idx].tolist()
        subject_groups.append(group)
else:
    group_size = len(df) // num_groups
    remainder = len(df) % num_groups
    start_idx = 0
    for i in range(num_groups):
        if i < remainder:
            end_idx = start_idx + group_size + 1
        else:
            end_idx = start_idx + group_size
        group = df['Subject'].iloc[start_idx:end_idx].tolist()
        subject_groups.append(group)
        start_idx = end_idx

roc = ROCCalculator(json_file_path)

# Calculate and print the range of image quality for each group
for i, group in enumerate(subject_groups):
    qualities = [image_quality[subject] for subject in group]
    quality_range = (min(qualities), max(qualities))
    auc = roc.calculate_ROC(group)['AUROC']
    print(f"Group {i+1} with length {len(group)} Image Quality Range: {quality_range} , AUC: {auc}")

