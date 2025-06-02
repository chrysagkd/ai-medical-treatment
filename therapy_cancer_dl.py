import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt

# --- Συνθετικά δεδομένα για θεραπεία καρκίνου ---
np.random.seed(42)
n_samples = 300

data = pd.DataFrame({
    'tumor_size': np.random.uniform(0.5, 5.0, n_samples),  # cm
    'lymph_nodes': np.random.randint(0, 10, n_samples),
    'age': np.random.randint(30, 85, n_samples),
    'genetic_marker': np.random.choice([0,1], n_samples),
    # target θεραπείας: 0 = Χημειοθεραπεία, 1 = Ακτινοθεραπεία, 2 = Χειρουργική επέμβαση
    'treatment': np.random.choice([0,1,2], n_samples, p=[0.4, 0.3, 0.3])
})

# --- Προετοιμασία ---
X = data.drop(columns='treatment').values
y = tf.keras.utils.to_categorical(data['treatment'], num_classes=3)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- Neural Network μοντέλο ---
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- Εκπαίδευση με αποθήκευση ιστορικού για οπτικοποίηση ---
history = model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)

# --- Αξιολόγηση ---
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Cancer Treatment Prediction - Test Accuracy: {accuracy:.2%}")

# --- Οπτικοποίηση Loss & Accuracy ---
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Train Loss')
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.title('Training Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# --- Πρόβλεψη για νέο ασθενή με οπτικοποίηση πιθανοτήτων ---
def predict_treatment(tumor_size, lymph_nodes, age, genetic_marker):
    features = np.array([[tumor_size, lymph_nodes, age, genetic_marker]])
    features = scaler.transform(features)
    pred_probs = model.predict(features)[0]
    pred_class = np.argmax(pred_probs)
    treatments = {0: 'Chemotherapy', 1: 'Radiation Therapy', 2: 'Surgery'}

    # Οπτικοποίηση πιθανοτήτων
    plt.figure(figsize=(6,4))
    plt.bar(treatments.values(), pred_probs, color=['red','blue','green'])
    plt.title('Predicted Treatment Probabilities')
    plt.ylabel('Probability')
    plt.ylim(0,1)
    for i, v in enumerate(pred_probs):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
    plt.show()

    return treatments.get(pred_class, "Unknown")

# --- Demo ---
if __name__ == "__main__":
    new_patient = {'tumor_size': 2.5, 'lymph_nodes': 3, 'age': 60, 'genetic_marker': 1}
    recommendation = predict_treatment(**new_patient)
    print(f"Recommended cancer treatment: {recommendation}")
