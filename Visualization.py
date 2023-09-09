import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_excel("Data1.xlsx", skiprows=1)
scaler = StandardScaler()

# Scale the data and store it in data_feat
selected_features = [col for col in data.columns if col not in ['Cf(mg/L)', 'AE(%)']]
scaler.fit(data[selected_features])

# Scale the data and store it in data_feat
scaled_features = scaler.transform(data[selected_features])
data_feat = pd.DataFrame(scaled_features, columns=selected_features)

# Create the pairplot with 'AC(mg/g)' as hue
sns.pairplot(data_feat, diag_kind='kde', hue='AC(mg/g)')
plt.show()
