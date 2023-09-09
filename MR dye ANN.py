import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor, MLPClassifier
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load your data
data = pd.read_excel("Data.xlsx", skiprows=1)

# Separate input features and output variables
X = data[['Stirringspeed', 'Temp', 'Time', 'Dosage', 'pH', 'Concentration']]
y = data[['Concentration,Cf(mg/L)', 'Adsorption capacity(mg/g)', 'Adsorption efficiency(%)']]
# Standardize the input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA to reduce dimensionality to 2 components
pca = PCA(n_components=6)
X_pca = pca.fit_transform(X_scaled)

X_pca_train, X_pca_test, y_pca_train, y_pca_test = train_test_split(X, y, test_size=0.10, random_state=42)
# Create and fit the model
model = MLPRegressor(hidden_layer_sizes=(20, 20), activation='relu', solver='adam', max_iter=200, tol=1e-4, validation_fraction=0.15)
model.fit(X_pca_train, y_pca_train)

# Make predictions on the test data
y_train = model.predict(X_pca_train)
y_pca_pred = model.predict(X_pca_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_pca_test, y_pca_pred)
print(y_pca_pred)
print("Mean Squared Error:", mse)
print("R2:", r2_score(y_pca_test, y_pca_pred, sample_weight=None, force_finite=True))

#result_df = pd.concat([X_pca_test.reset_index(drop=True), pd.DataFrame(y_pca_pred, columns=['Predicted_Output1', 'Predicted_Output2', 'Predicted_Output3'])], axis=1)

# Export the combined data to an Excel file
#result_df.to_excel('PCA predicted_data_with_inputs.xlsx', index=False)