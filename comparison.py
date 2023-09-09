#import pandas as pd
#import matplotlib.pyplot as plt

# Create a DataFrame with your R2 and MSE data
#data = {
   # "Model": ["ANN", "Random Forest", "Gradient Boosting", "SVRrbf", "SVRpoly", "Decision Tree", "XGBoost", "K-Neighbors"],
   # "Train R2": [0.999999657635581, 1.00000000000000, 0.99999324234707, 0.95186772710643, 0.36653608802279, 1.0, 0.9999, 0.998927571],
   # "Test R2": [0.99999965763558, 1.00000000000000, 0.99999324234707, 0.95186772710643, 0.36653608802279, 1.00000000000000, 0.99999999999728, 0.998927571],
   #"Train MSE": [0.00059082742122, 0.000000000000001, 0.39148456472997, 26.74378544936130, 351.97222016277700, 0.000000000000001, 0.00000004317097, 35.55442558],
   # "Test MSE": [0.00059082742122, 0.000000000000001, 0.39148456472997, 0.95186772710643, 0.36653608802279, 0.000000000000001, 0.00000004317097, 35.55442558]}

#results_df = pd.DataFrame(data)

# Create separate subplots for R2 and MSE
#fig, axes = plt.subplots(1,2, figsize=(12, 10))

# Plot Training MSE
#axes[0].barh(results_df["Model"], results_df["Train MSE"], color='darkcyan')
#axes[0].set_xlabel('Mean Squared Error (MSE)')
#axes[0].set_title('Training MSE Scores')

# Plot Testing MSE
#axes[1].barh(results_df["Model"], results_df["Test MSE"], color='darkblue')
#axes[1].set_xlabel('Mean Squared Error (MSE)')
#axes[1].set_title('Testing MSE Scores')

#plt.tight_layout()
#plt.show()#

import pandas as pd
import matplotlib.pyplot as plt

# Create a DataFrame with your R2 and MSE data
data = {
    "Model": ["ANN", "Random Forest", "Gradient Boosting", "SVRrbf", "SVRpoly", "Decision Tree", "XGBoost",
              "K-Neighbors"],
    "Train R2": [0.999999657635581, 1.00000000000000, 0.99999324234707, 0.95186772710643, 0.36653608802279, 1.0, 0.9999,
                 0.998927571],
    "Test R2": [0.99999965763558, 1.00000000000000, 0.99999324234707, 0.95186772710643, 0.36653608802279,
                1.00000000000000, 0.99999999999728, 0.998927571],
    "Train MSE": [0.00059082742122, 0.000000000000001, 0.39148456472997, 26.74378544936130, 351.97222016277700,
                  0.000000000000001, 0.00000004317097, 35.55442558],
    "Test MSE": [0.00059082742122, 0.000000000000001, 0.39148456472997, 0.95186772710643, 0.36653608802279,
                 0.000000000000001, 0.00000004317097, 35.55442558]}

results_df = pd.DataFrame(data)

# List of model names
models = results_df["Model"]

# Create a tailored plot for each model
for model_name in models:
    # Filter data for the current model
    model_data = results_df[results_df["Model"] == model_name]

    # Create separate subplots for R2 and MSE
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot Training R2
    axes[0].barh(model_data["Model"], model_data["Train R2"], color='darkcyan')
    axes[0].set_xlabel('R2 Score')
    axes[0].set_title('Training R2 Scores')

    # Plot Testing R2
    axes[1].barh(model_data["Model"], model_data["Test R2"], color='darkblue')
    axes[1].set_xlabel('R2 Score')
    axes[1].set_title('Testing R2 Scores')

    plt.tight_layout()
    plt.suptitle(f'Tailored Plot for {model_name}')
    plt.subplots_adjust(top=0.85)  # Adjust the title position
    plt.show()
