import pandas as pd

df = pd.read_csv("../dataset_titanic/train.csv")
print(df.head(2518))

nan_matrix = df.isnull()
print(nan_matrix.sum())

df['Cabin'] = df['Cabin'].fillna(df['Cabin'].mode()[0], inplace=False)
df['Age'] =df['Age'].fillna(df['Age'].median(), inplace=False)
df['RoomService'] =df['RoomService'].fillna(df['RoomService'].mean(), inplace=False)
df['HomePlanet'] =df['HomePlanet'].fillna(df['HomePlanet'].mode()[0], inplace=False)
df['CryoSleep'] =df['CryoSleep'].fillna(df['CryoSleep'].median(), inplace=False)
df['FoodCourt'] =df['FoodCourt'].fillna(df['FoodCourt'].mean(), inplace=False)
df['ShoppingMall'] =df['ShoppingMall'].fillna(df['ShoppingMall'].mode()[0], inplace=False)
df['Spa'] =df['Spa'].fillna(df['Spa'].median(), inplace=False)
df['VRDeck'] =df['VRDeck'].fillna(df['VRDeck'].mean(), inplace=False)
df['Destination'] =df['Destination'].fillna(df['Destination'].mode()[0], inplace=False)
df['VIP'] =df['VIP'].fillna(df['VIP'].mean(), inplace=False)
df['Name'] =df['Name'].fillna(df['Name'].mode()[0], inplace=False)
nan_matrix = df.isnull()
print(nan_matrix.sum())

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
numeric_df = df.select_dtypes(include='number')
for col in numeric_df:
    scaler.fit(df[[col]])
    df[col] = scaler.transform(df[[col]])

df = pd.get_dummies(df, columns=['Destination'], drop_first=True)

df.to_csv("../dataset_titanic/dataprocessed_titanic.csv", index=False)
