import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("../dataset_titanic/dataprocessed_titanic.csv")

X = df.drop(['VRDeck'], axis=1)
y = df['VRDeck']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.4, random_state=42)

from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()
linear_model.fit(X_train[['Age']], y_train)
y_pred_test = linear_model.predict(X_test[['Age']])

from sklearn.metrics import root_mean_squared_error
RMSE = root_mean_squared_error(y_test, y_pred_test)
print("RMSE = ", RMSE)

from sklearn.metrics import mean_absolute_error
MAE = mean_absolute_error(y_test, y_pred_test)
print("MAE = ", MAE)

import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize=(12, 8))
plt.scatter(X['Age'], y, alpha=0.6, color='blue', label='Фактические данные')
age_range = np.linspace(X['Age'].min(), X['Age'].max(), 100)
age_range_df = pd.DataFrame(age_range, columns=['Age'])
y_range = linear_model.predict(age_range_df)

plt.plot(age_range, y_range, color='red', linewidth=2, label='Линия регрессии')
plt.xlabel('Age', fontsize=12)
plt.ylabel('VRDeck', fontsize=12)
plt.title('VRDeck(Age)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


print("\n" + "=" * 500 + "\n")


X = df.drop(['Transported'], axis=1)
y = df['Transported']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.4, random_state=42)

features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

from sklearn.linear_model import LogisticRegression
logreg_model = LogisticRegression()

logreg_model.fit(X_train[features], y_train)
y_pred_test = logreg_model.predict(X_test[features])

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_test)
print(cm)

from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred_test)
print(report)