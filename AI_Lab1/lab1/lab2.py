import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("../dataset_titanic/dataprocessed_titanic.csv")

X = df.drop(['Age'], axis=1)
y = df['Age']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.4, random_state=42)

from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_test = linear_model.predict(X_test)