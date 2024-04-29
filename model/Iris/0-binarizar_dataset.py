import pandas as pd
import numpy as np
import os
  



url = os.path.dirname(os.path.abspath(__file__)) + "\\"


dataset_original = url + "Iris.csv" 
dataset_binarizado = url + "Iris_binarizado.csv"
coluna_target = "Species"


df = pd.read_csv(dataset_original)




#Remover linhas com valore ausentes
print("DF SEM DADOS AUSENTES")
#print(df.info())

df = df.drop("Id", axis=1)
df = df.dropna()




X = df.drop(coluna_target, axis=1)
y = df[coluna_target]


#Para SepalLengthCm #####################################
min_value = df['SepalLengthCm'].min()
max_value = df['SepalLengthCm'].max()
num_bins = 4
prefixo = "SepalLengthCm"
bin_width = (max_value - min_value) / num_bins
bin_edges = np.linspace(min_value, max_value + bin_width, num_bins + 1)
bins_labels = [prefixo+"_"+str(x) for x in range(num_bins)]
df['SepalLengthCm_label'] = pd.cut(df['SepalLengthCm'], bins=bin_edges, labels=bins_labels)
dummies_sepalLengthCm = pd.get_dummies(df['SepalLengthCm_label'], drop_first=False)
#print(dummies_sepalLengthCm)




#Para SepalWidthCm #####################################
min_value = df['SepalWidthCm'].min()
max_value = df['SepalWidthCm'].max()
num_bins = 4
prefixo = "SepalWidthCm"
bin_width = (max_value - min_value) / num_bins
bin_edges = np.linspace(min_value, max_value + bin_width, num_bins + 1)
bins_labels = [prefixo+"_"+str(x) for x in range(num_bins)]
df['SepalWidthCm_label'] = pd.cut(df['SepalWidthCm'], bins=bin_edges, labels=bins_labels)
dummies_SepalWidthCm = pd.get_dummies(df['SepalWidthCm_label'], drop_first=False)
#print(dummies_sepalLengthCm)




#Para PetalLengthCm #####################################
min_value = df['PetalLengthCm'].min()
max_value = df['PetalLengthCm'].max()
num_bins = 4
prefixo = "PetalLengthCm"
bin_width = (max_value - min_value) / num_bins
bin_edges = np.linspace(min_value, max_value + bin_width, num_bins + 1)
bins_labels = [prefixo+"_"+str(x) for x in range(num_bins)]
df['PetalLengthCm_label'] = pd.cut(df['PetalLengthCm'], bins=bin_edges, labels=bins_labels)
dummies_PetalLengthCm = pd.get_dummies(df['PetalLengthCm_label'], drop_first=False)
#print(dummies_sepalLengthCm)


#Para PetalWidthCm #####################################
min_value = df['PetalWidthCm'].min()
max_value = df['PetalWidthCm'].max()
num_bins = 4
prefixo = "PetalWidthCm"
bin_width = (max_value - min_value) / num_bins
bin_edges = np.linspace(min_value, max_value + bin_width, num_bins + 1)
bins_labels = [prefixo+"_"+str(x) for x in range(num_bins)]
df['PetalWidthCm_label'] = pd.cut(df['PetalWidthCm'], bins=bin_edges, labels=bins_labels)
dummies_PetalWidthCm = pd.get_dummies(df['PetalWidthCm_label'], drop_first=False)
#print(dummies_sepalLengthCm)





df_final = pd.concat([dummies_sepalLengthCm, dummies_SepalWidthCm, dummies_PetalLengthCm, dummies_PetalWidthCm, y], axis=1)


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




