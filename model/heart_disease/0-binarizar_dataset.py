import pandas as pd
import numpy as np
import os
  



url = os.path.dirname(os.path.abspath(__file__)) + "\\"


dataset_original = url + "heart_disease.csv" 
dataset_binarizado = "heart_disease_binarizado.csv"
coluna_target = "num"



df = pd.read_csv(dataset_original)

#Remover linhas com valore ausentes
print("DF SEM DADOS AUSENTES")
print(df.info())


X = df.drop("num", axis=1)
y = df["num"]


#Para age
cut_points = [0, 30, 60, 120]
bin_labels = ['age_young', 'age_adult', 'age_elderly']
df['age_label'] = pd.cut(df['age'], bins=cut_points, labels=bin_labels)
dummies_age = pd.get_dummies(df['age_label'], drop_first=False)


#Para sex
dummies_sex = pd.get_dummies(df['sex'], prefix="sex").replace({1: True, 0: False})



#Para cp  (chest pain)
dummies_cp = pd.get_dummies(df['cp'], prefix="cp").replace({1: True, 0: False})


#Para trestbps
cut_points = [0, 130, 1000]
bin_labels = ['trestbps_13', 'trestbps_14']
df['trestbps_label'] = pd.cut(df['trestbps'], bins=cut_points, labels=bin_labels)
dummies_trestbps = pd.get_dummies(df['trestbps_label'], drop_first=False)


#Para chol
cut_points = [0, 180, 1000]
bin_labels = ['chol_baixo', 'chol_alto']
df['chol'] = pd.cut(df['chol'], bins=cut_points, labels=bin_labels)
dummies_chol = pd.get_dummies(df['chol'], drop_first=False)


#Para fbs  (já é True, False.)
dummies_fbs = pd.get_dummies(df['fbs'], prefix="fbs", drop_first=False)


#Para restecg
dummies_restecg = pd.get_dummies(df['restecg'], prefix="restecg").replace({1: True, 0: False})

#Para thalach
cut_points = [0, 121, 1000]
bin_labels = ['thalach_baixo', 'thalach_alto']
df['thalach_'] = pd.cut(df['thalach'], bins=cut_points, labels=bin_labels)
dummies_thalach = pd.get_dummies(df['thalach_'], drop_first=False)


#Para exang (já é True False)
dummies_exang = pd.get_dummies(df['exang'], prefix="exang", drop_first=False)


#Para oldpeak
median_value = df['oldpeak'].median()
dummies_oldpeak = pd.DataFrame()
dummies_oldpeak['oldpeak_1'] = df['oldpeak'] < median_value
dummies_oldpeak['oldpeak_2'] = df['oldpeak'] >= median_value



#Para slope
dummies_slope = pd.get_dummies(df['slope'], prefix="slope").replace({1: True, 0: False})


#Para ca
dummies_ca = pd.get_dummies(df['ca'], prefix="ca").replace({1: True, 0: False})

#Para thal
dummies_thal = pd.get_dummies(df['thal'], prefix="thal").replace({1: True, 0: False})



df_final = pd.concat([dummies_age, dummies_sex, dummies_cp, dummies_trestbps, dummies_chol, dummies_fbs, dummies_restecg, dummies_thalach, dummies_exang, dummies_oldpeak, dummies_slope, dummies_ca, dummies_thal, y], axis=1)
print("Tamanho antes de remover duplicadas: ", len(df_final))

colunas = df_final.columns[:-1]

duplicates = df_final.duplicated(subset=colunas, keep=False)

df_duplicates = df_final[duplicates]


df_final = df_final.drop(df_duplicates.index)



print("Tamanho depois de remover duplicadas: ", len(df_final))
df_final.to_csv(dataset_binarizado, index=False)




print(df_final)
    
