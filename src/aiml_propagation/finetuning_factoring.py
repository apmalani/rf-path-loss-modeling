import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('src/aiml_propagation/combined_output_data.csv')

## step 4.5: factors
x = data.drop('gamma_ideal', axis = 1)
y = data['gamma_ideal']

rf = RandomForestRegressor()
rf.fit(x, y)

importances = rf.feature_importances_

fts = pd.DataFrame({'feature': x.columns, 'weight': importances})
fts = fts.sort_values('weight', ascending = False)

plt.figure(figsize = (12, 6))
sns.barplot(x = "weight", y = "feature", data = fts)
plt.title('Feature Importances')
plt.show()