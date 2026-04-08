import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load datasets
cab = pd.read_csv("cab_rides.csv")
weather = pd.read_csv("weather.csv")

# Drop rows where price is missing
cab = cab.dropna(subset=["price"])

# Convert timestamps
cab["datetime"] = pd.to_datetime(cab["time_stamp"], unit="ms")
weather["datetime"] = pd.to_datetime(weather["time_stamp"], unit="s")

# Create time-based features
cab["hour"] = cab["datetime"].dt.hour
cab["day_of_week"] = cab["datetime"].dt.dayofweek

# Sort before merging
cab = cab.sort_values("datetime")
weather = weather.sort_values("datetime")

# Merge cab rides with weather data
merged = pd.merge_asof(
    cab,
    weather,
    left_on="datetime",
    right_on="datetime",
    left_by="source",
    right_by="location",
    direction="nearest",
    tolerance=pd.Timedelta("1h")
)

print("Merged shape:", merged.shape)
print("\nMissing values after merge:")
print(merged.isnull().sum())

# Sample data to speed up training
merged = merged.sample(n=100000, random_state=42)

# Keep ride features + only temperature and rain for weather
features = [
    "distance",
    "cab_type",
    "destination",
    "source",
    "surge_multiplier",
    "name",
    "hour",
    "day_of_week",
    "temp",
    "rain"
]

target = "price"

merged = merged[features + [target]].copy()

X = merged[features]
y = merged[target]

# Separate numeric and categorical features
numeric_features = [
    "distance",
    "surge_multiplier",
    "hour",
    "day_of_week",
    "temp",
    "rain"
]

categorical_features = [
    "cab_type",
    "destination",
    "source",
    "name"
]

# Preprocessing
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

# Models
lr_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

rf_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=50, random_state=42))
])

# Train models
print("\nTraining Linear Regression...")
lr_model.fit(X_train, y_train)

print("Training Random Forest...")
rf_model.fit(X_train, y_train)

# Evaluation function
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return mae, rmse, r2

# Evaluate models
lr_mae, lr_rmse, lr_r2 = evaluate_model(lr_model, X_test, y_test)
rf_mae, rf_rmse, rf_r2 = evaluate_model(rf_model, X_test, y_test)

# Results table
results = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "MAE": [lr_mae, rf_mae],
    "RMSE": [lr_rmse, rf_rmse],
    "R2": [lr_r2, rf_r2]
})

print("\nModel Comparison:")
print(results)

# User input for prediction
distance = float(input("\nEnter distance: "))
cab_type = input("Enter cab type (Uber/Lyft): ")
source = input("Enter source: ")
destination = input("Enter destination: ")
surge = float(input("Enter surge multiplier: "))
name = input("Enter ride type (UberX, Lyft, Shared, Lux, etc.): ")
hour = int(input("Enter hour (0-23): "))
day = int(input("Enter day of week (0=Mon, 6=Sun): "))
temp = float(input("Enter temperature: "))
rain = float(input("Enter rain: "))

new_ride = pd.DataFrame([{
    "distance": distance,
    "cab_type": cab_type,
    "destination": destination,
    "source": source,
    "surge_multiplier": surge,
    "name": name,
    "hour": hour,
    "day_of_week": day,
    "temp": temp,
    "rain": rain
}])

print("\nNew ride data:")
print(new_ride)

print("\nMaking prediction...")
predicted_price = rf_model.predict(new_ride)

print("\nPredicted Price:", predicted_price[0])