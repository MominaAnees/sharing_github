import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data
data = pd.read_excel("Data1.xlsx", skiprows=1)

# Split data into inputs (X) and outputs (y)
X = data.drop(columns=['Stirringspeed', 'Temp', 'Time', 'Dosage', 'pH', 'Concentration'], axis=1)
y = data[['Concentration,Cf(mg/L)', 'Adsorption capacity(mg/g)', 'Adsorption efficiency(%)']]
X.columns = X.columns.astype(str)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Create the SVM regressor
svr = SVR(kernel='poly', C=1.0, gamma='scale', epsilon=0.1)

# Create the multi-output regressor
model = MultiOutputRegressor(svr)

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate Mean Squared Error for each output
mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
print("Mean Squared Error for Output 1:", mse[0])
print("Mean Squared Error for Output 2:", mse[1])
print("Mean Squared Error for Output 3:", mse[2])

# Calculate R2 Score for each output
r2 = r2_score(y_test, y_pred, multioutput='raw_values')
print("R2 Score for Output 1:", r2[0])
print("R2 Score for Output 2:", r2[1])
print("R2 Score for Output 3:", r2[2])
#result_df = pd.concat([X_test.reset_index(drop=True), pd.DataFrame(y_pred, columns=['Predicted_Output1', 'Predicted_Output2', 'Predicted_Output3'])], axis=1)

# Exporting the combined data to an Excel file
#result_df.to_excel('predicted_data_with_inputs_SVM.xlsx', index=False)

#df = pd.read_excel("predicted_data_with_inputs_SVM.xlsx")
#sns.set_theme(style='darkgrid')
# Create subplots in one line
#sns.lmplot(x='Concentration,Cf(mg/L)', y ='predicted_Concentration,Cf(mg/L)', data = df)
#sns.lmplot(x='Adsorption capacity(mg/g)', y ='predicted_Adsorption capacity(mg/g)', data = df)
#sns.lmplot(x='Adsorption efficiency(%)', y ='predicted_Adsorption efficiency(%)', data = df)
#plt.tight_layout()
#plt.show()


