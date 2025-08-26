import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow import keras
from keras import layers

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("Week 5 Project GNCIPL/archive (11).zip")

# Drop Loan_ID if present
if "Loan_ID" in df.columns:
    df.drop("Loan_ID", axis=1, inplace=True)

# -----------------------------
# 2. Preprocessing 
# -----------------------------
for col in df.columns:
    if df[col].dtype == "object":
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

# Handle 'Dependents' column (convert '3+' -> 3)
if "Dependents" in df.columns:
    df["Dependents"] = df["Dependents"].replace("3+", 3).astype(int)

# Convert categorical → one-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Features & Target
X = df.drop("Loan_Status_Y", axis=1)
y = df["Loan_Status_Y"]

# ✅ Ensure no NaN left
print("Any NaN left?", X.isnull().sum().sum())

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# 3. Handle Imbalance with SMOTE
# -----------------------------
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)

# -----------------------------
# 4. Build Improved ANN Model
# -----------------------------
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Learning rate schedule
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=1000,
    decay_rate=0.9
)

optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# -----------------------------
# 5. Train with EarlyStopping
# -----------------------------
callback = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=8, restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[callback],
    verbose=1
)

# -----------------------------
# 6. Evaluate Model
# -----------------------------
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Not Approved","Approved"],
            yticklabels=["Not Approved","Approved"])
plt.title("Confusion Matrix (Improved ANN)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -----------------------------
# 7. Training Graphs
# -----------------------------
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label="Train Acc")
plt.plot(history.history['val_accuracy'], label="Val Acc")
plt.xlabel("Epochs"); plt.ylabel("Accuracy")
plt.title("Accuracy over Epochs"); plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.xlabel("Epochs"); plt.ylabel("Loss")
plt.title("Loss over Epochs"); plt.legend()

plt.show()

# -----------------------------
# 8. Predict New Application
# -----------------------------
def predict_loan(sample_dict):
    sample_df = pd.DataFrame([sample_dict])
    sample_df = pd.get_dummies(sample_df)

    # Align columns with training data
    sample_df = sample_df.reindex(columns=X.columns, fill_value=0)

    sample_scaled = scaler.transform(sample_df)
    pred = model.predict(sample_scaled)[0][0]
    return "Approved ✅" if pred > 0.5 else "Not Approved ❌"

# Example
sample_app = {
    "ApplicantIncome": 5000,
    "CoapplicantIncome": 2000,
    "LoanAmount": 150,
    "Loan_Amount_Term": 360,
    "Credit_History": 1,
    "Dependents": 2,
    "Gender": "Male",
    "Married": "Yes",
    "Education": "Graduate",
    "Self_Employed": "No",
    "Property_Area": "Urban"
}

print("\n🔮 Loan Prediction:", predict_loan(sample_app))
