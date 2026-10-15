import numpy as np
import pandas as pd
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

np.random.seed(42)

data_size = 500

data = pd.DataFrame({
    "study_hours": np.random.uniform(1, 10, data_size),
    "attendance": np.random.uniform(50, 100, data_size),
    "previous_score": np.random.uniform(30, 100, data_size),
    "sleep_hours": np.random.uniform(4, 10, data_size)
})

noise = np.random.normal(0, 5, data_size)

data["final_score"] = (
    data["study_hours"] * 4
    + data["attendance"] * 0.2
    + data["previous_score"] * 0.3
    + data["sleep_hours"] * 1.5
    + noise
)

data["final_score"] = data["final_score"].clip(0, 100)

X = data[[
    "study_hours",
    "attendance",
    "previous_score",
    "sleep_hours"
]]

y = data["final_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))

with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model trained and saved successfully!")
