import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load data from Excel
data = pd.read_excel("Data.xlsx", skiprows=1)


# Split data into inputs (X) and outputs (y)
X = data.drop(columns=['Stirringspeed', 'Temp', 'Time', 'Dosage', 'pH', 'Concentration'], axis=1)
y = data[['Concentration,Cf(mg/L)', 'Adsorption capacity(mg/g)', 'Adsorption efficiency(%)']]

# Convert column names to strings
X.columns = X.columns.astype(str)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Create and fit the RandomForestRegressor model
model = RandomForestRegressor(n_estimators=200, max_features=6,bootstrap=True, max_depth=None, oob_score=True, random_state=42)

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate Mean Squared Error and R2
mse = mean_squared_error(y_test, y_pred)
print(y_pred)
print("Mean Squared Error:", mse)
print("R2:", r2_score(y_test, y_pred, sample_weight=None, force_finite=True))

# Combine input values and predicted values into one DataFrame
#result_df = pd.concat([X_test.reset_index(drop=True), pd.DataFrame(y_pred, columns=['Predicted_Output1', 'Predicted_Output2', 'Predicted_Output3'])], axis=1)

# Export the combined data to an Excel file
#result_df.to_excel('predicted_data_with_inputs_RF.xlsx', index=False)

# Export the combined data to an Excel file
#result_df.to_excel('predicted_data_with_inputs.xlsx', index=False)


# Load the combined data with inputs and predicted values
#df = pd.read_excel('predicted_data_with_inputs_RF.xlsx')

# Set seaborn style
#sns.set_theme(style='darkgrid')

# Load the combined data with inputs and predicted values
#df = pd.read_excel('predicted_data_with_inputs_RF.xlsx')

# Create a regression plot for Output1, output 2, and output 3
#sns.lmplot(x='Concentration,Cf(mg/L)', y ='predicted_Concentration,Cf(mg/L)', data = df)
#sns.lmplot(x='Adsorption capacity(mg/g)', y ='predicted_Adsorption capacity(mg/g)', data = df)
#sns.lmplot(x='Adsorption efficiency(%)', y ='predicted_Adsorption efficiency(%)', data = df)

# Use tight_layout to ensure proper spacing
#plt.tight_layout()

# Show the plot
#plt.show()

# Extract experiment numbers, Output1 values, and predicted Output1 values
#experiment_numbers = df['Experiment'][:200]
#output1_values = df['Concentration,Cf(mg/L)'][:200]
#predicted_output1_values = df['predicted_Concentration,Cf(mg/L)'][:200]

# Create a plot
#plt.figure(figsize=(10, 6))
#plt.plot(experiment_numbers, output1_values, label='Experiment', linestyle='-', color = 'blue')
#plt.plot(experiment_numbers, predicted_output1_values, label='RF', linestyle='-', color = 'red', linewidth=1)

# Set labels and title
#plt.xlabel('Experiment Number')
#plt.ylabel('Concentration,Cf(mg/L)')
#plt.title('Experimental vs. RF Predicted values')

# Add legend
#plt.legend()

# Show the plot
#plt.grid()
#plt.show()