from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib

# Load Patient Data
df = pd.read_csv('patient_data.csv')

# 3. Prepare features and labels
X = df.drop('diagnosis', axis=1)  # Features
y = df['diagnosis']                # Labels

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Scale features (important for medical data!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 7. Evaluate
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

# 8. Save model
joblib.dump(model, 'disease_model.pkl')
joblib.dump(scaler, 'scaler.pkl')