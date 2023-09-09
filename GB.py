import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data
data = pd.read_excel("Data.xlsx", skiprows=1)

# Split data into inputs (X) and outputs (y)
X = data.drop(columns=['Stirringspeed', 'Temp', 'Time', 'Dosage', 'pH', 'Concentration'], axis=1)
y = data[['Concentration,Cf(mg/L)', 'Adsorption capacity(mg/g)', 'Adsorption efficiency(%)']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Create the Gradient Boosting Regressor model
# You can adjust hyperparameters like n_estimators, learning_rate, max_depth, etc.
model = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
)

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate Mean Squared Error for each output
#mse = mean_squared_error(y_test, y_pred)
#print("Mean Squared Error:", mse)

# Calculate R2 Score for each output
#r2 = r2_score(y_test, y_pred)
#print("R2 Score:", r2)

#y_train_pred = model.predict(X_train)

# Make predictions on the testing data
#y_test_pred = model.predict(X_test)
#r2_train = r2_score(y_train, y_train_pred)
#r2_test = r2_score(y_test, y_test_pred)
#mse_train = mean_squared_error(y_train, y_train_pred)
#mse_test = mean_squared_error(y_test, y_test_pred)
#print("MSE for Training:", mse_train)
#print("MSE for Testing", mse_test)
#print("R2 Score for Training:", r2_train)
#print("R2 Score for Testing:", r2_test)


 #Combine input values and predicted values into one DataFrame
#result_df = pd.concat([X_test.reset_index(drop=True), pd.DataFrame(y_pred, columns=['Predicted_Output1', 'Predicted_Output2', 'Predicted_Output3'])], axis=1)

# Export the combined data to an Excel file
#result_df.to_excel('predicted_data_with_inputs_GB.xlsx', index=False)


# Export the combined data to an Excel file
#result_df.to_excel('predicted_data_with_inputs_GB.xlsx', index=False)

# Set seaborn style
sns.set_theme(style='darkgrid')

# Load the combined data with inputs and predicted values
df = pd.read_excel('predicted_data_with_inputs_GB.xlsx')

# Create a regression plot for Output1, output 2, and output 3
sns.lmplot(x='Concentration,Cf(mg/L)', y ='Predicted_Concentration,Cf(mg/L)', data = df)
sns.lmplot(x='Adsorption capacity(mg/g)', y ='Predicted_Adsorption capacity(mg/g)', data = df)
sns.lmplot(x='Adsorption efficiency(%)', y ='Predicted_Adsorption efficiency(%)', data = df)

# Use tight_layout to ensure proper spacing
plt.tight_layout()

# Show the plot
plt.show()


