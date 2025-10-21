import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

# Load dataset
df = pd.read_csv('crop_yield.csv')

# Define features and target
features = [
    'Ph', 'K', 'P', 'N', 'Zn', 'S',
    'QV2M-W', 'QV2M-Sp', 'QV2M-Su', 'QV2M-Au',
    'T2M_MAX-W', 'T2M_MAX-Sp', 'T2M_MAX-Su', 'T2M_MAX-Au',
    'T2M_MIN-W', 'T2M_MIN-Sp', 'T2M_MIN-Su', 'T2M_MIN-Au',
    'PRECTOTCORR-W', 'PRECTOTCORR-Sp', 'PRECTOTCORR-Su', 'PRECTOTCORR-Au',
    'WD10M', 'GWETTOP', 'CLOUD_AMT', 'WS2M_RANGE', 'PS'
]
X = df[features]
y = df['crop_yield_tph']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train XGBoost regressor
model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"\n🌾 XGBoost Regression Results")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Scatter plot with diagonal reference line and square aspect ratio
plt.figure(figsize=(8, 8))
sns.scatterplot(x=y_test, y=y_pred, color='mediumseagreen', alpha=0.6, edgecolor='black')

# Diagonal line (perfect prediction)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], color='gray', linestyle='--', linewidth=2)

# Set equal aspect ratio
plt.axis('square')
plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)

# Labels and title
plt.xlabel("Actual Crop Yield (tph)", fontsize=12)
plt.ylabel("Predicted Crop Yield (tph)", fontsize=12)
plt.title("Actual vs Predicted Crop Yield", fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()

# Feature importance
importances = model.feature_importances_
top_indices = np.argsort(importances)[-10:]
top_features = [features[i] for i in top_indices]
top_scores = importances[top_indices]

plt.figure(figsize=(8, 5))
sns.barplot(x=top_scores, y=top_features, palette='viridis')
plt.title("Top 10 Influential Features")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()
