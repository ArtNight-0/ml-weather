# example: https://www.scirp.org/journal/paperinformation?paperid=132019

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Sample data - Load economic data from a CSV or database
data = {
    'gdp_per_capita': [167640, 90176, 47691, 47334, 68357, 57713, 45753, 36518, 72619, 57051, 127285, 110450, 60763, 55061, 73297, 60391, 49206, 110506, 96138, 159385, 44671, 80027, 58029, 50474, 54285, 67545, 34059, 49455, 56445, 54684, 58557],
    'urbanization_rate': [86.50, 83.15, 56.43, 58.41, 62.71, 68.10, 56.98, 59.55, 61.18, 51.71, 69.61, 68.90, 54.69, 56.00, 60.30, 56.02, 47.52, 65.82, 70.70, 88.13, 50.22, 65.50, 52.29, 47.69, 31.14, 58.13, 47.69, 54.47, 58.88, 50.91, 59.06]
}
df = pd.DataFrame(data)

# Features and labels
X = df['gdp_per_capita']
y = df['urbanization_rate']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train.values.reshape(-1, 1), y_train)

# Predict something
print(model.predict([[150000]]))

# Hasil: [83.27521968]
# JSON: 83.27521968317754
# Web app: Predicted Urbanization Rate: 83.27521968317754%

# Save model
joblib.dump(model, 'urbanization_growth_model.pkl')
