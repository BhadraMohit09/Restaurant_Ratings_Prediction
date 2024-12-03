# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = '/mnt/data/Dataset .csv'
dataset = pd.read_csv('Dataset .csv')

# Step 1: Preprocess the dataset
# Drop irrelevant columns
columns_to_drop = ['Restaurant ID', 'Restaurant Name', 'Address', 'Locality', 
                   'Locality Verbose', 'Rating color', 'Rating text']
dataset_cleaned = dataset.drop(columns=columns_to_drop)

# Handle missing values
dataset_cleaned['Cuisines'] = dataset_cleaned['Cuisines'].fillna('Unknown')

# Encode categorical variables using one-hot encoding
categorical_columns = ['City', 'Cuisines', 'Currency', 
                       'Has Table booking', 'Has Online delivery', 
                       'Is delivering now', 'Switch to order menu']
dataset_encoded = pd.get_dummies(dataset_cleaned, columns=categorical_columns, drop_first=True)

# Separate features and target variable
X = dataset_encoded.drop(columns=['Aggregate rating'])
y = dataset_encoded['Aggregate rating']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Train the regression model
regressor = DecisionTreeRegressor(random_state=42)
regressor.fit(X_train, y_train)

# Step 3: Evaluate the model
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")

# Step 4: Interpret the model's results
# Analyze the most influential features
feature_importances = pd.Series(regressor.feature_importances_, index=X_train.columns)
top_features = feature_importances.sort_values(ascending=False).head(10)

print("\nTop 10 Influential Features:")
print(top_features)
