import pandas as pd
import numpy as np
import os
  



url = os.path.dirname(os.path.abspath(__file__)) + "\\"


dataset_original = url + "breast-cancer.csv" 
dataset_binarizado = url + "breast_cancer_data_binarizado.csv"
coluna_target = "diagnosis"







#Abrir dataset original
df = pd.read_csv(dataset_original)



#print(df)

#Remover linhas com valore ausentes
#print("DF SEM DADOS AUSENTES")
#print(df.info())
#exit()

#Remover coluna Id caso exista
colunas_a_remover = ["id","radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"]
df = df.drop(colunas_a_remover, axis=1)

print(df)


#Remover linhas com valores ausentes
df = df.dropna()
df = df.drop_duplicates()

#Separar X e y
X = df.drop(coluna_target, axis=1)
y = df[coluna_target]



#Para radius_mean #####################################
min_value = df['radius_mean'].min()
max_value = df['radius_mean'].max()
num_bins = 4
prefixo = "radius_mean"
bin_width = (max_value - min_value) / num_bins
bin_edges = np.linspace(min_value, max_value + bin_width, num_bins + 1)
bins_labels = [prefixo+"_"+str(x) for x in range(num_bins)]
df['radius_mean_label'] = pd.cut(df['radius_mean'], bins=bin_edges, labels=bins_labels)
dummies_radius_mean = pd.get_dummies(df['radius_mean_label'], drop_first=False)
#print(dummies_sepalLengthCm)




#Para texture_mean #####################################
min_value = df['texture_mean'].min()
max_value = df['texture_mean'].max()
num_bins = 4
prefixo = "texture_mean"
bin_width = (max_value - min_value) / num_bins
bin_edges = np.linspace(min_value, max_value + bin_width, num_bins + 1)
bins_labels = [prefixo+"_"+str(x) for x in range(num_bins)]
df['texture_mean_label'] = pd.cut(df['texture_mean'], bins=bin_edges, labels=bins_labels)
dummies_texture_mean = pd.get_dummies(df['texture_mean_label'], drop_first=False)
#print(dummies_sepalLengthCm)




#Para compactness_mean #####################################
min_value = df['compactness_mean'].min()
max_value = df['compactness_mean'].max()
num_bins = 5
prefixo = "compactness_mean"
bin_width = (max_value - min_value) / num_bins
bin_edges = np.linspace(min_value, max_value + bin_width, num_bins + 1)
bins_labels = [prefixo+"_"+str(x) for x in range(num_bins)]
df['compactness_mean_label'] = pd.cut(df['compactness_mean'], bins=bin_edges, labels=bins_labels)
dummies_Compactness = pd.get_dummies(df['compactness_mean_label'], drop_first=False)
#print(dummies_sepalLengthCm)


#Para perimeter_mean #####################################
min_value = df['perimeter_mean'].min()
max_value = df['perimeter_mean'].max()
num_bins = 4
prefixo = "perimeter_mean"
bin_width = (max_value - min_value) / num_bins
bin_edges = np.linspace(min_value, max_value + bin_width, num_bins + 1)
bins_labels = [prefixo+"_"+str(x) for x in range(num_bins)]
df['perimeter_mean_label'] = pd.cut(df['perimeter_mean'], bins=bin_edges, labels=bins_labels)
dummies_perimeter_mean = pd.get_dummies(df['perimeter_mean_label'], drop_first=False)
#print(dummies_sepalLengthCm)







#Para area_mean #####################################
min_value = df['area_mean'].min()
max_value = df['area_mean'].max()
num_bins = 4
prefixo = "area_mean"
bin_width = (max_value - min_value) / num_bins
bin_edges = np.linspace(min_value, max_value + bin_width, num_bins + 1)
bins_labels = [prefixo+"_"+str(x) for x in range(num_bins)]
df['area_mean_label'] = pd.cut(df['area_mean'], bins=bin_edges, labels=bins_labels)
dummies_area_mean = pd.get_dummies(df['area_mean_label'], drop_first=False)
#print(dummies_sepalLengthCm)





#Para smoothness_mean #####################################
min_value = df['smoothness_mean'].min()
max_value = df['smoothness_mean'].max()
num_bins = 4
prefixo = "smoothness_mean"
bin_width = (max_value - min_value) / num_bins
bin_edges = np.linspace(min_value, max_value + bin_width, num_bins + 1)
bins_labels = [prefixo+"_"+str(x) for x in range(num_bins)]
df['smoothness_mean_label'] = pd.cut(df['smoothness_mean'], bins=bin_edges, labels=bins_labels)
dummies_smoothness_mean = pd.get_dummies(df['smoothness_mean_label'], drop_first=False)
#print(dummies_sepalLengthCm)




#Para compactness_mean #####################################
min_value = df['compactness_mean'].min()
max_value = df['compactness_mean'].max()
num_bins = 4
prefixo = "compactness_mean"
bin_width = (max_value - min_value) / num_bins
bin_edges = np.linspace(min_value, max_value + bin_width, num_bins + 1)
bins_labels = [prefixo+"_"+str(x) for x in range(num_bins)]
df['compactness_mean_label'] = pd.cut(df['compactness_mean'], bins=bin_edges, labels=bins_labels)
dummies_compactness_mean = pd.get_dummies(df['compactness_mean_label'], drop_first=False)
#print(dummies_sepalLengthCm)




#Para concavity_mean #####################################
min_value = df['concavity_mean'].min()
max_value = df['concavity_mean'].max()
num_bins = 5
prefixo = "concavity_mean"
bin_width = (max_value - min_value) / num_bins
bin_edges = np.linspace(min_value, max_value + bin_width, num_bins + 1)
bins_labels = [prefixo+"_"+str(x) for x in range(num_bins)]
df['concavity_mean_label'] = pd.cut(df['concavity_mean'], bins=bin_edges, labels=bins_labels)
dummies_concavity_mean = pd.get_dummies(df['concavity_mean_label'], drop_first=False)
#print(dummies_sepalLengthCm



#Para concave points_mean #####################################
min_value = df['concave points_mean'].min()
max_value = df['concave points_mean'].max()
num_bins = 5
prefixo = "concave points_mean"
bin_width = (max_value - min_value) / num_bins
bin_edges = np.linspace(min_value, max_value + bin_width, num_bins + 1)
bins_labels = [prefixo+"_"+str(x) for x in range(num_bins)]
df['concave points_mean_label'] = pd.cut(df['concave points_mean'], bins=bin_edges, labels=bins_labels)
dummies_points_mean = pd.get_dummies(df['concave points_mean_label'], drop_first=False)
#print(dummies_sepalLengthCm




#Para symmetry_mean #####################################
min_value = df['symmetry_mean'].min()
max_value = df['symmetry_mean'].max()
num_bins = 5
prefixo = "symmetry_mean"
bin_width = (max_value - min_value) / num_bins
bin_edges = np.linspace(min_value, max_value + bin_width, num_bins + 1)
bins_labels = [prefixo+"_"+str(x) for x in range(num_bins)]
df['symmetry_mean_label'] = pd.cut(df['symmetry_mean'], bins=bin_edges, labels=bins_labels)
dummies_symmetry_mean = pd.get_dummies(df['symmetry_mean_label'], drop_first=False)
#print(dummies_sepalLengthCm



#Para fractal_dimension_mean #####################################
min_value = df['fractal_dimension_mean'].min()
max_value = df['fractal_dimension_mean'].max()
num_bins = 5
prefixo = "fractal_dimension_mean"
bin_width = (max_value - min_value) / num_bins
bin_edges = np.linspace(min_value, max_value + bin_width, num_bins + 1)
bins_labels = [prefixo+"_"+str(x) for x in range(num_bins)]
df['fractal_dimension_mean_label'] = pd.cut(df['fractal_dimension_mean'], bins=bin_edges, labels=bins_labels)
dummies_fractal_dimension_mean = pd.get_dummies(df['fractal_dimension_mean_label'], drop_first=False)
#print(dummies_sepalLengthCm




df_final = pd.concat([dummies_radius_mean, dummies_texture_mean, dummies_Compactness, dummies_perimeter_mean, dummies_area_mean, dummies_smoothness_mean, dummies_compactness_mean, dummies_concavity_mean, dummies_points_mean, dummies_symmetry_mean, dummies_fractal_dimension_mean, y], axis=1)





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




