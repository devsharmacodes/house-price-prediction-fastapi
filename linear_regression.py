# Import libraries
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("house_prices_practice.csv")

# Features and target
X = data.drop(["SalePrice", "Id"], axis=1)
y = data['SalePrice']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Save model
with open("house_price_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully!")