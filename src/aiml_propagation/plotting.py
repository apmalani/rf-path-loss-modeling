import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

results = pd.read_csv('results/training_predicts_v_actuals.csv')

sns.set(style="whitegrid")

plt.figure(figsize=(12, 8))
scatter_plot = sns.scatterplot(data=results, x='actual', y='predicted', hue='city', palette='deep', s=50, edgecolor = None)
sns.regplot(data=results, x='actual', y='predicted', scatter=False, color='red')

plt.title('Actual vs Predicted by City', fontsize=16)
plt.xlabel('Actual Values', fontsize=14)
plt.ylabel('Predicted Values', fontsize=14)

plt.legend(title='City', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)

plt.tight_layout()
plt.show()
