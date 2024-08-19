import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
from scipy.stats import mannwhitneyu
from roc_calculator import ROCCalculator
import random

'''# Replace 'your_file.xlsx' with the path to your Excel file
file_path = '/Volumes/pelvis/projects/tiago/iqa/marksheet/the_marksheet.xlsx'

# Read the Excel sheet into a DataFrame
df = pd.read_excel(file_path)

# Extract the 3rd, 4th, and 5th columns (columns are 0-indexed, so columns 2, 3, and 4)
selected_columns = df.iloc[:, [2, 3, 4]]

cleaned_df = selected_columns.dropna()'''


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

# Sample with replacement
groups = []

for i, group in enumerate(subject_groups):
    aucs = []
    print(f'Group length {len(group)}')
    while len(aucs) < 10000:
        n = len(group)
        k = int(random.uniform(2, n))
        sampled_strings = random.choices(group, k = k)
        # Check if all targets are the same
        targets = [case_target[s] for s in sampled_strings]
        if all(t == 0 for t in targets) or all(t == 1 for t in targets):
            continue
        auc = roc.calculate_ROC(sampled_strings)['AUROC']
        aucs.append(auc)
    groups.append(aucs)

# Assuming groups is a list of lists, where each inner list contains AUC values for a group
# groups = [group1_aucs, group2_aucs, ..., group5_aucs]

'''# Extract AUC values for Group 1 and Group 5
group1_aucs = groups[0]

group5_aucs = groups[4]

# Perform the Mann-Whitney U test
stat, p_value = mannwhitneyu(group1_aucs, group5_aucs, alternative='greater')

# Print the results
print(f'Mann-Whitney U test statistic: {stat}')
print(f'P-value: {p_value}')

# Interpret the results
alpha = 0.05
if p_value < alpha:
    print('Group 1 AUC values are significantly higher than Group 5 AUC values.')
else:
    print('There is no significant difference or Group 5 AUC values are higher.')
'''


# Assuming groups is a list of lists, where each inner list contains AUC values for a group
# groups = [group1_aucs, group2_aucs, ...]

# Create a box plot
fig, ax = plt.subplots()
ax.boxplot(groups, tick_labels=[f'Group {i+1}' for i in range(len(groups))])

# Add titles and labels
ax.set_title('Bootstrapped AUC Distribution by Group')
ax.set_xlabel('Groups')
ax.set_ylabel('AUC')

# Show the plot
plt.show()

# Create a histogram plot for Group 1 and Group 5
fig, ax = plt.subplots()
ax.hist(groups[0], bins=100, alpha=0.3, color='yellow', label='Group 1')
#ax.hist(groups[1], bins=100, alpha=0.3, color='green', label='Group 2')
#ax.hist(groups[2], bins=100, alpha=0.3, color='blue', label='Group 3')
#ax.hist(groups[3], bins=100, alpha=0.5, color='orange', label='Group 4')
ax.hist(groups[4], bins=100, alpha=0.5, color='red', label='Group 5')

# Add titles and labels for the histogram plot
ax.set_title('Overlapped Histograms of Group 1 and Group 5 AUCs')
ax.set_xlabel('AUC')
ax.set_ylabel('Frequency')
ax.legend()

# Show the histogram plot
plt.show()

# Calculate and print the range of image quality for each group
for i, group in enumerate(subject_groups):
    qualities = [image_quality[subject] for subject in group]
    quality_range = (min(qualities), max(qualities))
    auc = roc.calculate_ROC(group)['AUROC']
    print(f"Group {i+1} with length {len(group)} Image Quality Range: {quality_range} , AUC: {auc}")

