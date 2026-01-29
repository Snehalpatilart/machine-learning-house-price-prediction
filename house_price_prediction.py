import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Step 1: Load Dataset
data = {
    "Area": [800, 1000, 1200, 1500, 1800, 2000],
    "Bedrooms": [1, 2, 2, 3, 3, 4],
    "Price": [50000, 65000, 72000, 90000, 105000, 120000]
}

df = pd.DataFrame(data)

# Step 2: Split Features and Target
X = df[["Area", "Bedrooms"]]
y = df["Price"]

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predictions
predictions = model.predict(X_test)

# Step 6: Evaluation
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)

print("Model Trained Successfully!")
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)

# Step 7: User Input Prediction
area = int(input("Enter house area in sqft: "))
bedrooms = int(input("Enter number of bedrooms: "))

predicted_price = model.predict([[area, bedrooms]])
print("Estimated House Price:", int(predicted_price[0]))
