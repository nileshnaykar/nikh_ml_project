
# House Price Prediction using Machine Learning

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# 1. Create a sample dataset for practice
np.random.seed(42)

n = 500

area = np.random.randint(500, 4000, n)
bedrooms = np.random.randint(1, 6, n)
bathrooms = np.random.randint(1, 5, n)
age = np.random.randint(0, 40, n)

# Generate sample prices in rupees
price = (
    area * 3500
    + bedrooms * 500000
    + bathrooms * 300000
    - age * 25000
    + np.random.normal(0, 200000, n)
)

price = np.maximum(price, 500000)

# 2. Store data in a Pandas DataFrame
df = pd.DataFrame({
    "Area": area,
    "Bedrooms": bedrooms,
    "Bathrooms": bathrooms,
    "Age": age,
    "Price": price
})

print("First 5 rows:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

# 3. Visualize the data
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="Area", y="Price")
plt.title("House Area vs House Price")
plt.xlabel("Area (sq. ft.)")
plt.ylabel("Price (Rs.)")
plt.tight_layout()
plt.show()

# 4. Correlation heatmap
plt.figure(figsize=(7, 5))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# 5. Select features and target
X = df[["Area", "Bedrooms", "Bathrooms", "Age"]]
y = df["Price"]

# 6. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Create and train the XGBoost model
model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    random_state=42,
    objective="reg:squarederror"
)

model.fit(X_train, y_train)

# 8. Predict prices on test data
y_pred = model.predict(X_test)

# 9. Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"MAE: Rs. {mae:,.2f}")
print(f"RMSE: Rs. {rmse:,.2f}")
print(f"R2 Score: {r2:.4f}")

# 10. Predict the price of a new house
new_house = pd.DataFrame({
    "Area": [1500],
    "Bedrooms": [3],
    "Bathrooms": [2],
    "Age": [5]
})

predicted_price = model.predict(new_house)[0]

print("\nNew House Details:")
print(new_house)

print(f"Predicted House Price: Rs. {predicted_price:,.2f}")

# 11. Compare actual and predicted prices
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual Price (Rs.)")
plt.ylabel("Predicted Price (Rs.)")
plt.title("Actual vs Predicted House Prices")

# Ideal prediction line
low = min(y_test.min(), y_pred.min())
high = max(y_test.max(), y_pred.max())
plt.plot([low, high], [low, high], "r--")

plt.tight_layout()
plt.show()

# 12. Feature importance
importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values()

importance.plot(kind="barh", figsize=(7, 4))
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()
