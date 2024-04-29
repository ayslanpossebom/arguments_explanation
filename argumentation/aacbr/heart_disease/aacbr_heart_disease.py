from src.aacbr import Aacbr, Case
import os
import pickle
import pandas as pd


url = os.path.dirname(os.path.abspath(__file__)) + "\\"
path_essential_arguments = url+"heart_arguments.ob"


#Carregar lista de argumentos

with open(path_essential_arguments, 'rb') as fp:
    arguments = pickle.load(fp)

print(f"Total of essential arguments: {len(arguments)}")


#Definindo instancias de teste
df = pd.read_csv(url+"heart_disease_binarizado.csv")
df = df.sample(5)
X = df.drop('num', axis=1)
y = df['num'].astype(int)
target_column = 'num'




expected_output = list(y)
results = []

#Múltiplas classes: executar uma vez para cada classe
unique_values = df[target_column].unique()
for value in unique_values:

    #open list with essential arguments to train the model
    cb = []

    default = Case('default', set(), outcome=value)
    cb.append(default)

    i = 1
    for arg in arguments:
        case = Case(f'case{i}', arg["premissas"], outcome=arg["conclusao"])
        cb.append(case)
        i += 1

    train_data = cb


    #Definindo casos de teste
    test_data = []

    for index, row in df.iterrows():
        features = [col for col in X.columns if row[col] == 1]
        case = Case(f'new{index}', set(features))
        test_data.append(case)



    #Training the model AACBR
    clf = Aacbr()

    clf.fit(train_data)

    predicted_output = clf.predict(test_data)

    results.append({"target": value, "result": predicted_output})


print("Right answers:        ", expected_output)
print("AACBR classification: ")
for line in results:
    print(f"Class {line['target']}: {line['result']}")


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

dtree = DecisionTreeClassifier()
df_train = pd.read_csv(url+"heart_disease_binarizado.csv")

Xdf = df_train.drop('num', axis=1)
ydf = df_train['num']
x_train, x_test, y_train, y_test = train_test_split(Xdf, ydf, test_size=0.3, random_state=42)

dtree.fit(x_train, y_train)
y_pred = dtree.predict(X)
print("Decision Tree classification: ", y_pred)

