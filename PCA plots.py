import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

# Load your data
data = pd.read_excel("Data.xlsx", skiprows=1)

# Separate input features and output variables
X = data[['Stirringspeed', 'Temp', 'Time', 'Dosage', 'pH', 'Concentration']]

# Standardize the input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA to reduce dimensionality to 2 components
pca = PCA(n_components=6)
X_pca = pca.fit_transform(X_scaled)

# Define labels for input variables
input_labels = ['Stirringspeed', 'Temp', 'Time', 'Dosage', 'pH', 'Concentration']

def biplot(score, coeff, labels=None, target_groups=None):
    xs = score[:, 0]
    ys = score[:, 1]
    n = coeff.shape[1]
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())

    plt.figure(figsize=(10, 8))

    # Plot the data points with colors based on target groups
    if target_groups is not None:
        sns.scatterplot(x=xs * scalex, y=ys * scaley, hue=target_groups, palette="viridis")
    else:
        plt.scatter(xs * scalex, ys * scaley, c='blue')

    if labels is not None:
        for i in range(n):
            plt.arrow(0, 0, coeff[0, i], coeff[1, i], head_width=0.03, head_length=0.03, color='red', alpha=0.5)
            plt.text(coeff[0, i] * 1.15, coeff[1, i] * 1.15, labels[i], fontsize=12)

# Call the biplot function with input labels
biplot(X_pca, np.transpose(pca.components_), labels=input_labels)
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.grid()
plt.show()