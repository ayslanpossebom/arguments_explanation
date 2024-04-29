import pandas as pd
import numpy as np
import os
  



url = os.path.dirname(os.path.abspath(__file__)) + "\\"


dataset_original = url + "titanic_train.csv" 
dataset_binarizado = url + "titanic_train_binarizado.csv"
coluna_target = "Survived"







#Abrir dataset original
df = pd.read_csv(dataset_original)

#print(df)

#Remover linhas com valore ausentes
#print("DF SEM DADOS AUSENTES")
#print(df.info())
#exit()

#Remover coluna Id caso exista
colunas_a_remover = ['PassengerId','Ticket', 'Name', 'Cabin']
df = df.drop(colunas_a_remover, axis=1)



#Remover linhas com valores ausentes
df = df.dropna()
df = df.drop_duplicates()

print(df)


#Separar X e y
X = df.drop(coluna_target, axis=1)
y = df[coluna_target]



#Para Pclass #####################################
dummies_pclass = pd.get_dummies(df['Pclass'], prefix='Pclass')
#print(dummies_pclass)




#Para Sex #####################################
dummies_sex = pd.get_dummies(df['Sex'], prefix='Sex')
#print(dummies_sex)





#Para Age #####################################
min_value = df['Age'].min()
max_value = df['Age'].max()
num_bins = 5
prefixo = "Age"
bin_width = (max_value - min_value) / num_bins
bin_edges_age = np.linspace(min_value, max_value + bin_width, num_bins + 1)
bins_labels_age = [prefixo+"_"+str(x) for x in range(num_bins)]
df['Age_label'] = pd.cut(df['Age'], bins=bin_edges_age, labels=bins_labels_age)
dummies_age = pd.get_dummies(df['Age_label'], drop_first=False)
#print(dummies_age)



#Para SibSp #####################################
dummies_Sibsp = pd.get_dummies(df['SibSp'], prefix='SibSp')
#print(df['SibSp'].unique())
#print(dummies_Sibsp)




#Para Fare #####################################
min_value = df['Fare'].min()
max_value = df['Fare'].max()
num_bins = 6
prefixo = "Fare"
bin_width = (max_value - min_value) / num_bins
bin_edges_fare = np.linspace(min_value, max_value + bin_width, num_bins + 1)
bins_labels_fare = [prefixo+"_"+str(x) for x in range(num_bins)]
df['Fare_label'] = pd.cut(df['Fare'], bins=bin_edges_fare, labels=bins_labels_fare)
dummies_Fare = pd.get_dummies(df['Fare_label'], drop_first=False)
#print(dummies_Fare)




#Para Parch #####################################
dummies_Parch = pd.get_dummies(df['Parch'], prefix='Parch')
#print(dummies_Parch)





#Para Embarked #####################################
dummies_Embarked = pd.get_dummies(df['Embarked'], prefix='Embarked')
#print(dummies_Embarked)





df_final = pd.concat([dummies_pclass, dummies_sex, dummies_age, dummies_Sibsp, dummies_Fare, dummies_Parch, dummies_Embarked, y], axis=1)





#Remover linhas duplicadas para DF ficar consistente
print("Tamanho atual: ", len(df_final))
#df_final.to_csv(dataset_binarizado+"temp", index=False)

df_final = df_final.drop_duplicates()
print("Tamanho depois de remover linhas duplicadas: ", len(df_final))

colunas = df_final.columns[:-1]
duplicates = df_final.duplicated(subset=colunas, keep=False)

df_duplicates = df_final[duplicates]
df_final = df_final.drop(df_duplicates.index)
print("Tamanho depois de remover duplicadas inconsistentes: ", len(df_final))






df_final.to_csv(dataset_binarizado, index=False)

print(df_final)







############ TESTES ########################
df = pd.read_csv("titanic_test.csv")
colunas_a_remover = ['PassengerId','Ticket', 'Name', 'Cabin']
df = df.drop(colunas_a_remover, axis=1)

dummies_pclass = pd.get_dummies(df['Pclass'], prefix='Pclass')
#print(dummies_pclass)

dummies_sex = pd.get_dummies(df['Sex'], prefix='Sex')
#print(dummies_sex)

#Para Age #####################################
df['Age_label'] = pd.cut(df['Age'], bins=bin_edges_age, labels=bins_labels_age)
dummies_age = pd.get_dummies(df['Age_label'], drop_first=False)
#print(dummies_age)

#Para SibSp #####################################
dummies_Sibsp = pd.get_dummies(df['SibSp'], prefix='SibSp')
#print(dummies_Sibsp)

#Para Fare #####################################
df['Fare_label'] = pd.cut(df['Fare'], bins=bin_edges_fare, labels=bins_labels_fare)
dummies_Fare = pd.get_dummies(df['Fare_label'], drop_first=False)
#print(dummies_Fare)

#Para Parch #####################################
dummies_Parch = pd.get_dummies(df['Parch'], prefix='Parch')
#print(dummies_Parch)

#Para Embarked #####################################
dummies_Embarked = pd.get_dummies(df['Embarked'], prefix='Embarked')
#print(dummies_Embarked)


df_final = pd.concat([dummies_pclass, dummies_sex, dummies_age, dummies_Sibsp, dummies_Fare, dummies_Parch, dummies_Embarked, y], axis=1)

df_final = df_final.dropna()
df_final = df_final.drop_duplicates()

df_final.to_csv("titanic_test_binarizado.csv", index=False)

