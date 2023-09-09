import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load your dataset
data = pd.read_excel("Data.xlsx", skiprows=1)

# Separate input features and output variables
X = data[['Stirringspeed', 'Temp', 'Time', 'Dosage', 'pH', 'Concentration']]
y = data['Concentration,Cf(mg/L)']  # Assuming you want to predict this target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create an ExtraTreesRegressor
extra_trees_regressor = ExtraTreesRegressor(n_estimators=100, random_state=42)  # You can adjust the number of trees (n_estimators)

# Fit the regressor to the training data
extra_trees_regressor.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = extra_trees_regressor.predict(X_test)

# Calculate mean squared error (MSE) and R-squared (R2)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R-squared (R2):", r2)
y_train_pred = extra_trees_regressor.predict(X_train)
y_test_pred = extra_trees_regressor.predict(X_test)
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
print("MSE for Training:", mse_train)
print("MSE for Testing", mse_test)
print("R2 Score for Training:", r2_train)
print("R2 Score for Testing:", r2_test)