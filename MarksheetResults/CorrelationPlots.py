from roc_calculator import ROCCalculator
import matplotlib.pyplot as plt
#, subject_list=hq_subset

path_to_json = '/Volumes/pelvis/projects/tiago/iqa/marksheet/the_json.json'

calculator = ROCCalculator(path_to_json)

roc_data = calculator.calculate_ROC()
calculator.calculate_ROC_by_quality()

'''plt.figure()
plt.plot(roc_data['FPR'], roc_data['TPR'], color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_data['AUROC'])
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()'''
print(calculator)
print(roc_data)
import numpy as np
import matplotlib.pyplot as plt

# Given data
y = np.array([0.793169255067123,0.8106756974681504, 0.8104541734860884,0.8127659574468085, 0.8316566063044937, 0.8359739049394221, 0.8519176136363638, 0.8744791666666667, 0.8744791666666667, 0.8744791666666667])
x = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])

coefficients = np.polyfit(x, y, 2)  # 2nd degree polynomial
polynomial = np.poly1d(coefficients)

correlation_coefficient = np.corrcoef(x, y)[0, 1]
print(correlation_coefficient)
# Create a range of x values for the trend line
x_fit = np.linspace(min(x), max(x), 400)

# Calculate the corresponding y values for the trend line
y_fit = polynomial(x_fit)

# Create the plot with a clean and modern style
plt.style.use('seaborn-v0_8-whitegrid')  # Set the style to a clean grid
plt.figure(figsize=(10, 6))

# Plot the trend line
plt.plot(x_fit, y_fit, color='#1f77b4', label='Trend Line')

# Plot the original data points
plt.scatter(x, y, color='#ff7f0e', label='Data Points')

# Add a horizontal line at y = 0.83
#plt.axhline(y=0.83, color='#2ca02c', linestyle='--', label='100  cases')

# Set labels and title
plt.xlabel('% of lowest-quality volumes removed')
plt.ylabel('Diagnostic AUC')
plt.title('Diagnostic AUC vs % of Lowest-Quality Volumes Removed')

# Add legend
plt.legend()

# Show the plot
plt.show()