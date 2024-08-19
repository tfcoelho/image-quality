import pandas as pd
import json

# Read the Excel file into a DataFrame
df = pd.read_excel('/Volumes/pelvis/projects/tiago/iqa/maarksheet/the_marksheet.xlsx')

# Filter out rows where 'Case_pred', 'GT', or 'IQ' is missing
df_filtered = df.dropna(subset=['Case_pred', 'GT', 'IQ'])

# Create the dictionaries
case_pred = df_filtered.set_index('Study')['Case_pred'].to_dict()
case_target = df_filtered.set_index('Study')['GT'].astype(int).to_dict()
image_quality = df_filtered.set_index('Study')['IQ'].to_dict()

# Combine the dictionaries into one
combined_dict = {
    'case_pred': case_pred,
    'case_target': case_target,
    'image_quality': image_quality
}

# Save the combined dictionary to a JSON file
with open('the_json.json', 'w') as json_file:
    json.dump(combined_dict, json_file, indent=4)
