import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

np.random.seed(42)
n_samples = 500

X = np.random.randint(0, 2, size=(n_samples, 12))

# Вся логика в одной строке (исправленная):
sums = np.sum(X, axis=1)
noise_mask = np.random.rand(n_samples) < 0.1  # 10% шума

# Если (сумма > 6) XOR (шум) → правящая партия, иначе оппозиция
# XOR: разные значения = 1, одинаковые = 0
condition = (sums > 6) != noise_mask  # Эквивалент XOR

y = np.zeros((n_samples, 2))
y[condition, 0] = 1  # Где условие True - правящая партия [1,0]
y[~condition, 1] = 1  # Где False - оппозиция [0,1]

# Для scikit-learn моделей: одномерные метки (0 или 1)
y_labels = np.argmax(y, axis=1)  # Преобразуем one-hot в метки классов

# Сохранение данных в файлы
np.savetxt('dataIn.txt', X, fmt='%d')
np.savetxt('dataOut.txt', y, fmt='%d')

# Разделяем на тренировочную и тестовую выборки
X_train, X_test, y_train_onehot, y_test_onehot, y_train_labels, y_test_labels = train_test_split(
    X, y, y_labels, test_size=0.3, random_state=42
)

# Нормализация данных
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = keras.Sequential([
   keras.layers.Dense(10, activation='relu', input_shape=(12,)),  # Скрытый слой 1
   keras.layers.Dense(8, activation='relu'),                      # Скрытый слой 2
   keras.layers.Dense(2, activation='softmax')                    # Выходной слой для one-hot
])

# Для one-hot encoding используем categorical_crossentropy
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Обучение модели (используем one-hot y)
history = model.fit(X_train_scaled, y_train_onehot, epochs=100, batch_size=10,
                   validation_data=(X_test_scaled, y_test_onehot))

# Оцениваем качество на тестовой выборке
y_pred_proba = model.predict(X_test_scaled)
y_pred_nn = np.argmax(y_pred_proba, axis=1)  # Преобразуем one-hot в метки
print("Neural Network Accuracy:", accuracy_score(y_test_labels, y_pred_nn))

# График изменения функции ошибки
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Function')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# Логистическая регрессия
lr = LogisticRegression()
lr.fit(X_train_scaled, y_train_labels)  # Используем одномерные метки
y_pred_lr = lr.predict(X_test_scaled)
print("Logistic Regression Accuracy:", accuracy_score(y_test_labels, y_pred_lr))

# Случайный лес
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train_labels)  # Используем одномерные метки
y_pred_rf = rf.predict(X_test_scaled)
print("Random Forest Accuracy:", accuracy_score(y_test_labels, y_pred_rf))