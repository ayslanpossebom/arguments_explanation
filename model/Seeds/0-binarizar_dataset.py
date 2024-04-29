import pandas as pd
import numpy as np
import os
  



url = os.path.dirname(os.path.abspath(__file__)) + "\\"


dataset_original = url + "seeds.csv" 
dataset_binarizado = url + "seeds_binarizado.csv"
coluna_target = "Type"







#Abrir dataset original
df = pd.read_csv(dataset_original)

#Remover linhas com valore ausentes
print("DF SEM DADOS AUSENTES")
print(df.info())


#Remover coluna Id caso exista
#df = df.drop("Id", axis=1)

#Remover linhas com valores ausentes
df = df.dropna()

#Separar X e y
X = df.drop(coluna_target, axis=1)
y = df[coluna_target]


#Para Area #####################################
min_value = df['Area'].min()
max_value = df['Area'].max()
num_bins = 5
prefixo = "Area"
bin_width = (max_value - min_value) / num_bins
bin_edges = np.linspace(min_value, max_value + bin_width, num_bins + 1)
bins_labels = [prefixo+"_"+str(x) for x in range(num_bins)]
df['Area_label'] = pd.cut(df['Area'], bins=bin_edges, labels=bins_labels)
dummies_Area = pd.get_dummies(df['Area_label'], drop_first=False)
#print(dummies_sepalLengthCm)




#Para Perimeter #####################################
min_value = df['Perimeter'].min()
max_value = df['Perimeter'].max()
num_bins = 5
prefixo = "Perimeter"
bin_width = (max_value - min_value) / num_bins
bin_edges = np.linspace(min_value, max_value + bin_width, num_bins + 1)
bins_labels = [prefixo+"_"+str(x) for x in range(num_bins)]
df['Perimeter_label'] = pd.cut(df['Perimeter'], bins=bin_edges, labels=bins_labels)
dummies_Perimeter = pd.get_dummies(df['Perimeter_label'], drop_first=False)
#print(dummies_sepalLengthCm)




#Para Compactness #####################################
min_value = df['Compactness'].min()
max_value = df['Compactness'].max()
num_bins = 5
prefixo = "Compactness"
bin_width = (max_value - min_value) / num_bins
bin_edges = np.linspace(min_value, max_value + bin_width, num_bins + 1)
bins_labels = [prefixo+"_"+str(x) for x in range(num_bins)]
df['Compactness_label'] = pd.cut(df['Compactness'], bins=bin_edges, labels=bins_labels)
dummies_Compactnessm = pd.get_dummies(df['Compactness_label'], drop_first=False)
#print(dummies_sepalLengthCm)


#Para Kernel.Length #####################################
min_value = df['Kernel.Length'].min()
max_value = df['Kernel.Length'].max()
num_bins = 5
prefixo = "Kernel.Length"
bin_width = (max_value - min_value) / num_bins
bin_edges = np.linspace(min_value, max_value + bin_width, num_bins + 1)
bins_labels = [prefixo+"_"+str(x) for x in range(num_bins)]
df['Kernel_Length_label'] = pd.cut(df['Kernel.Length'], bins=bin_edges, labels=bins_labels)
dummies_Kernel = pd.get_dummies(df['Kernel_Length_label'], drop_first=False)
#print(dummies_sepalLengthCm)







#Para Kernel.Width #####################################
min_value = df['Kernel.Width'].min()
max_value = df['Kernel.Width'].max()
num_bins = 5
prefixo = "Kernel.Width"
bin_width = (max_value - min_value) / num_bins
bin_edges = np.linspace(min_value, max_value + bin_width, num_bins + 1)
bins_labels = [prefixo+"_"+str(x) for x in range(num_bins)]
df['Kernel.Width_label'] = pd.cut(df['Kernel.Width'], bins=bin_edges, labels=bins_labels)
dummies_Kernel_Width = pd.get_dummies(df['Kernel.Width_label'], drop_first=False)
#print(dummies_sepalLengthCm)





#Para Asymmetry.Coeff #####################################
min_value = df['Asymmetry.Coeff'].min()
max_value = df['Asymmetry.Coeff'].max()
num_bins = 5
prefixo = "Asymmetry.Coeff"
bin_width = (max_value - min_value) / num_bins
bin_edges = np.linspace(min_value, max_value + bin_width, num_bins + 1)
bins_labels = [prefixo+"_"+str(x) for x in range(num_bins)]
df['Asymmetry.Coeff_label'] = pd.cut(df['Asymmetry.Coeff'], bins=bin_edges, labels=bins_labels)
dummies_Asymmetry_Coeff = pd.get_dummies(df['Asymmetry.Coeff_label'], drop_first=False)
#print(dummies_sepalLengthCm)




#Para Kernel.Groove #####################################
min_value = df['Kernel.Groove'].min()
max_value = df['Kernel.Groove'].max()
num_bins = 5
prefixo = "Kernel.Groove"
bin_width = (max_value - min_value) / num_bins
bin_edges = np.linspace(min_value, max_value + bin_width, num_bins + 1)
bins_labels = [prefixo+"_"+str(x) for x in range(num_bins)]
df['Kernel.Groove_label'] = pd.cut(df['Kernel.Groove'], bins=bin_edges, labels=bins_labels)
dummies_Kernel_Groove = pd.get_dummies(df['Kernel.Groove_label'], drop_first=False)
#print(dummies_sepalLengthCm)




df_final = pd.concat([dummies_Area, dummies_Perimeter, dummies_Compactnessm, dummies_Kernel, dummies_Kernel_Width, dummies_Asymmetry_Coeff, dummies_Kernel_Groove, y], axis=1)





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

