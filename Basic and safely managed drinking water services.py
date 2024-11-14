# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import joblib  # For saving the model and scaler
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Load the dataset from Google Drive
file_path = '/content/drive/My Drive/Basic and safely managed drinking water services.csv'
data = pd.read_csv(file_path)

# Preprocess the data
data_cleaned = data.dropna(subset=['Display Value']).copy()

# Encode categorical columns
label_encoder = LabelEncoder()
data_cleaned['WHO region'] = label_encoder.fit_transform(data_cleaned['WHO region'])
data_cleaned['Country'] = label_encoder.fit_transform(data_cleaned['Country'])
data_cleaned['Residence Area Type'] = label_encoder.fit_transform(data_cleaned['Residence Area Type'])

# Define features and target variable
X = data_cleaned[['Year', 'WHO region', 'Country', 'Residence Area Type']]
y = data_cleaned['Display Value']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply SMOTE to handle any class imbalance (for regression this is applied to the feature space)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train the Gradient Boosting Regressor model
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

# Save the trained model and scaler to Google Drive
joblib.dump(model, '/content/drive/My Drive/best_model.pkl')
joblib.dump(scaler, '/content/drive/My Drive/scaler.pkl')

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Print evaluation metrics
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)

# Display the top 10 predictions vs actual values
comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).head(10)
print(comparison_df)