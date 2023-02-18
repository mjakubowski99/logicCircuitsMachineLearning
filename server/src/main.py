from sklearn.preprocessing import LabelBinarizer
from preprocessing.to_binary import to_binary
from preprocessing.missing_values_adder import fill_missing
from preprocessing.tag_objects import tag_objects
from preprocessing.encode_decimal_to_integers import encode_decimal_to_integers
from preprocessing.to_binary import to_binary
from classifier.LogicClassifier import LogicClassifier
from sklearn.model_selection import train_test_split


import pandas as pd

data_set_url = "../datasets/dataset3/county_results.csv"
target = "Result"

df = pd.read_csv(data_set_url)
df[target] = df[target].astype(bool)

df.dropna(subset=[target], inplace=True)

df = fill_missing(df, target)
df = tag_objects(df, target)
df = encode_decimal_to_integers(df, target)
df = to_binary(df, target)

df.dropna(subset=[target], inplace=True)
y = df[target]
X = df.drop(columns=[target]).to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = LogicClassifier()
clf.fit(X_train, y_train)

print("Wynik dla danych testowych: ", clf.score(X_test,y_test))








