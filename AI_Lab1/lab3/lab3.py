import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("../dataset_titanic/dataprocessed_titanic.csv")
features = ['RoomService','FoodCourt','ShoppingMall','Spa']

X = df.drop(['VRDeck'], axis=1)
y = df['VRDeck']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.4, random_state=42)

from sklearn.tree import DecisionTreeRegressor
dt_regressor_model = DecisionTreeRegressor(max_depth=8, max_leaf_nodes = 10)
dt_regressor_model.fit(X_train[features], y_train)
y_pred_test = dt_regressor_model.predict(X_test[features])

from sklearn.metrics import root_mean_squared_error
RMSE = root_mean_squared_error(y_test, y_pred_test)
print("RMSE = ", RMSE)

from sklearn.metrics import mean_absolute_error
MAE = mean_absolute_error(y_test, y_pred_test)
print("MAE = ", MAE)

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
plot_tree(dt_regressor_model, feature_names=features, filled=True, rounded=True, fontsize=8, proportion = False)
plt.title("Граф дерева")
plt.show()


print("\n" + "=" * 500 + "\n")


X = df.drop(['Transported'], axis=1)
y = df['Transported']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.4, random_state=42)

from sklearn.tree import DecisionTreeClassifier
dt_classifier_model = DecisionTreeClassifier(max_depth=4, max_leaf_nodes = 10)
dt_classifier_model.fit(X_train[features], y_train)
y_pred_test = dt_classifier_model.predict(X_test[features])

y_proba = dt_classifier_model.predict_proba(X_test[features])
#print(dt_classifier_model.classes_)

from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba[:, 1])

plt.plot(fpr, tpr, color='red', marker='o')
plt.ylim([0,1.004])
plt.xlim([0,1.004])
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.title('ROC curve')
plt.grid(True, linestyle='--', alpha=0.7)
plt.plot([0,1.004], [0,1.004], linestyle='--', color='red', alpha=0.5)
plt.show()

from sklearn.metrics import auc
auc_metric = auc(fpr, tpr)
print("auc_metric = ",auc_metric)