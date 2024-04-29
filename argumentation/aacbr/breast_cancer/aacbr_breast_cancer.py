from src.aacbr import Aacbr, Case
import os
import pickle
import pandas as pd


url = os.path.dirname(os.path.abspath(__file__)) + "\\"
path_essential_arguments = url+"breast_arguments.ob"


#Carregar lista de argumentos

with open(path_essential_arguments, 'rb') as fp:
    arguments = pickle.load(fp)

print(f"Total of essential arguments: {len(arguments)}")


#Definindo instancias de teste
df = pd.read_csv(url+"breast_cancer_data_binarizado.csv")
df = df.sample(5)
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']
target_column = 'diagnosis'

expected_output = list(y)



#open list with essential arguments to train the model
cb = []

default = Case('default', set(), outcome='M')
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

