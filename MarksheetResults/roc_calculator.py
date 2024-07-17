import json
from typing import Optional, List, Dict, Any
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt

class ROCCalculator:
    def __init__(self, json_file_path: str):
        # Load the JSON data
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)

        # Extract the dictionaries
        self.case_target = data['case_target']
        self.case_pred = data['case_pred']
        self.image_quality = data['image_quality']

        # Create a list of subjects from the keys of 'case_target'
        self.subject_list = list(self.case_target.keys())

    def calculate_ROC(self, subject_list: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate Receiver Operating Characteristic curve for case-level risk stratification.
        """
        if subject_list is None:
            subject_list = self.subject_list

        # Ensure that all subjects in the subject_list have corresponding 'case_target' and 'case_pred' values
        valid_subjects = [s for s in subject_list if s in self.case_target and s in self.case_pred]

        fpr, tpr, _ = roc_curve(
            y_true=[self.case_target[s] for s in valid_subjects],
            y_score=[self.case_pred[s] for s in valid_subjects],
        )

        auroc = auc(fpr, tpr)

        return {
            'FPR': fpr.tolist(),  # Convert numpy arrays to lists for JSON serialization
            'TPR': tpr.tolist(),
            'AUROC': auroc,
        }

    def calculate_ROC_by_quality(self):
        # Sort subjects by image quality
        sorted_subjects = sorted(self.subject_list, key=lambda s: self.image_quality[s])
        high_quality = []
        # Calculate ROC for the 90% best quality and the 10% worst quality
        for i in range(0, 100, +10):
            threshold = np.percentile([self.image_quality[s] for s in sorted_subjects], i)
            print('threshold' + str(threshold))
            high_quality_subjects = [s for s in sorted_subjects if self.image_quality[s] >= threshold]
            #low_quality_subjects = [s for s in sorted_subjects if self.image_quality[s] < threshold]
            print(len(high_quality_subjects)/len(sorted_subjects))
            #print(len(low_quality_subjects))
            high_quality_roc = self.calculate_ROC(high_quality_subjects)
            #low_quality_roc = self.calculate_ROC(low_quality_subjects)
            high_quality.append(high_quality_roc)
            print(f"{100- i}% Best Quality AUROC: {high_quality_roc['AUROC']}")
            #print(f"{i}% Worst Quality AUROC: {low_quality_roc['AUROC']}")

            if i == 0:
                # Plot ROC curve
                plt.plot(high_quality_roc['FPR'], high_quality_roc['TPR'],
                         label=f"Full Set AUROC: {np.round(high_quality_roc['AUROC'], 2)}")
            else:
                # Plot ROC curve
                plt.plot(high_quality_roc['FPR'], high_quality_roc['TPR'],
                         label=f"Removing {i}% Lowest Quality AUROC: {np.round(high_quality_roc['AUROC'], 2)}")

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()

        return high_quality
