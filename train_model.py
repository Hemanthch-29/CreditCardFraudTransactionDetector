import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pickle
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
df = pd.read_csv(r"Cleaned_fraud_detection_transaction.csv")

# Select relevant features
feature_columns = [
    'Transaction_Amount', 'Transaction_Type', 'Device_Type', 'Is_Weekend',
    'Previous_Fraudulent_Activity', 'Daily_Transaction_Count',
    'Avg_Transaction_Amount_7d', 'Failed_Transaction_Count_7d',
    'Card_Type', 'Card_Age', 'Transaction_Distance',
    'Authentication_Method', 'Risk_Score'
]
X = df[feature_columns]
y = df['Fraud_Label']

# Encode categorical columns if needed
X = pd.get_dummies(X, columns=['Transaction_Type', 'Device_Type', 'Card_Type', 'Authentication_Method'])

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train SVM model
model = SVC(probability=True)
model.fit(X_train, y_train)
# Predict and evaluate
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# Save model and scaler
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
# Save model input columns (after get_dummies)
pickle.dump(X.columns.tolist(), open("model_columns.pkl", "wb"))
