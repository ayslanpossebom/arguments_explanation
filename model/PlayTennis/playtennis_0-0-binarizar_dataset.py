import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import os



url = os.path.dirname(os.path.abspath(__file__)) + "\\"


#PARAMETROS
dataset_original = "PlayTennis.csv"
dataset_binarizado = "PlayTennis_binarizado.csv"


df = pd.read_csv(url+dataset_original)
print(df)



#Binarizar dataset:

colunas = df.columns[:-1]
dummies_df = pd.get_dummies(df[colunas], drop_first=False)
dummies_df['Play Tennis'] = df['Play Tennis']

print(dummies_df.columns)


print(dummies_df)



#Remover linhas duplicadas para DF ficar consistente
print("Tamanho atual: ", len(dummies_df))

dummies_df = dummies_df.drop_duplicates()
print("Tamanho depois de remover linhas duplicadas: ", len(dummies_df))

colunas = dummies_df.columns[:-1]
duplicates = dummies_df.duplicated(subset=colunas, keep=False)

df_duplicates = dummies_df[duplicates]
df_final = dummies_df.drop(df_duplicates.index)
print("Tamanho depois de remover duplicadas inconsistentes: ", len(dummies_df))



dummies_df.to_csv(url+dataset_binarizado, index=False)
                        
