import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('crop_yield.csv')
df = df.dropna()

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
y = df['label']  # Crop type

# Encode crop labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)

# Train logistic regression model
model = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🌾 Logistic Regression Classification Results")
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion matrix
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=le.classes_, cmap='Blues')
plt.title("Confusion Matrix: Crop Type")
plt.tight_layout()
plt.show()
