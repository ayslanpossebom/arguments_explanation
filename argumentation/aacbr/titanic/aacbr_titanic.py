from src.aacbr import Aacbr, Case
import os
import pickle
import pandas as pd


url = os.path.dirname(os.path.abspath(__file__)) + "\\"
path_essential_arguments = url+"titanic_arguments.ob"


#Carregar lista de argumentos

with open(path_essential_arguments, 'rb') as fp:
    arguments = pickle.load(fp)

print(f"Total of essential arguments: {len(arguments)}")


#Definindo instancias de teste
df = pd.read_csv(url+"titanic_test_binarizado.csv")
df = df.sample(5)
X = df.drop('Survived', axis=1)
y = df['Survived'].astype(int)
target_column = 'Survived'

expected_output = list(y)



#open list with essential arguments to train the model
cb = []

default = Case('default', set(), outcome=0)
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


print("Right answers:        ", expected_output)
print("AACBR classification: ", predicted_output)



from sklearn.neighbors import KNeighborsClassifier
df_train = pd.read_csv(url+"titanic_train_binarizado.csv")
Xt = df_train.drop('Survived', axis=1)
yt = df_train['Survived']
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(Xt, yt)
X = X.drop(['Parch_9', 'SibSp_8'], axis=1)
knnprediction = knn.predict(X)
print("Knn Results:          ", knnprediction.tolist())

