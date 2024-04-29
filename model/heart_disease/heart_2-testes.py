import pandas as pd
from itertools import combinations
from collections import Counter
import os

url = os.path.dirname(os.path.abspath(__file__)) + "\\"


#CONFIGURAÇÃO
quantia_para_teste = 5




argumentos_heart = url + 'heart_argumentos_todos.ob'
#argumentos_heart = url + 'heart_argumentos.ob'


dataset_original = url + "heart_disease.csv" 
dataset_binarizado = url + "heart_disease_binarizado.csv"
dataset_binarizado_amostras = url + "heart_disease_binarizado_amostras_todos.csv"
#dataset_binarizado_amostras = url + "heart_disease_binarizado_amostras.csv"
coluna_target = "num"



#Realizar testes

df = pd.read_csv(dataset_binarizado)
resultados_possiveis = df[coluna_target].unique()




#Fatos do usuário
df_temp = pd.read_csv("df_temp.csv")



lista_geral = []
for i, row in df_temp.iterrows():
    atributos_com_1 = [col for col in df_temp.columns if col != coluna_target and row[col] == 1]
    premissa = set(atributos_com_1)
    print("TESTANDO ARGUMENTO ", premissa, " COM CONCLUSÃO ", row[coluna_target])
    lista_justificativa = []     


    for i in range(1, len(atributos_com_1)):                    

        combinacoes = combinations(atributos_com_1, i)

        #Para cada combinação possível
        for combinacao in combinacoes:
            #print("COMBINAÇÃO: ", combinacao)
            temp = set(combinacao)

            condicao = (df[list(temp)] == True).all(axis=1)
                
            df_filtrado = df.loc[condicao]
            valores_unicos = df_filtrado[coluna_target].unique()
                
            #se tiver valor único de resposta               
            if(len(valores_unicos) == 1):
                #print("A combinação gera resultado único")


                x = [conjunto for conjunto in lista_justificativa if conjunto["premissas"].issubset(temp)]
                #print("Conjuntos com interseção: ", len(x))
                if(len(x) > 0):
                    #print("Não pode incluir")
                    pass
                else:
                    #print("Pode incluir")
                    arg = {"premissas": temp, "conclusao": row[coluna_target]}
                    lista_justificativa.append(arg) 
                     
                continue 
        
    lista_geral.append(lista_justificativa)

X_train = df.drop(columns=[coluna_target], axis=1)
y_train = df[coluna_target]

X_test = df_temp.drop(columns=[coluna_target], axis=1)
y_test = df_temp[coluna_target]


from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
X_train = df.drop([coluna_target], axis=1)
y_train = df[[coluna_target]]

X_test = df_temp.drop([coluna_target], axis=1)
y_test = df_temp[[coluna_target]]

# 3. Treinar os modelos
# Árvore de Decisão
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)
# K-Nearest Neighbors (KNN)
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
knn_predictions = knn_model.predict(X_test)
# Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_predictions = nb_model.predict(X_test)
# 4. Avaliar e exibir os resultados

print("Árvore de Decisão    ", dt_predictions)
print("K-Nearest Neighbors  ", knn_predictions)
print("Naive Bayes          ", nb_predictions)

for i in range(0, len(lista_geral)):
    just = lista_geral[i]

    resposta = dt_predictions[i]
    print("Resposta: ", resposta)

    lista = []
    tamanho = 0
    for item in just:
        if(item["conclusao"] == resposta):
            if(len(item["premissas"]) > tamanho):
                lista = []
                lista.append(item)
                tamanho = len(item["premissas"])
            else:
                lista.append(item)

    """
    lista2 = []
    tamanho = 1000
    for item in just:
        if(item["conclusao"] == resposta):
            if(len(item["premissas"]) < tamanho):
                lista = []
                lista.append(item)
                tamanho = len(item["premissas"])
            elif(len(item["premissas"]) == tamanho):
                lista.append(item)
    """

    for aaa in lista:
        print("-   <", aaa["premissas"], ",",aaa["conclusao"], ">")
    print("")


conjuntoRespostas = []
ctjr = []
for argsjust in lista_geral:
    argsjust_conclusoes = [arg["conclusao"] for arg in argsjust]
    resultado = {opcao: 0 for opcao in df[coluna_target].unique()}
    resultado.update(Counter(argsjust_conclusoes))         
    conjuntoRespostas.append(resultado)

    maior_chave = max(resultado, key=resultado.get)
            
    maior_valor = max(resultado.values())
    maiores_chaves = [chave for chave, valor in resultado.items() if valor == maior_valor]
    ctjr.append(str(maiores_chaves))

fim = pd.DataFrame()


df_temporario = pd.DataFrame({
    'MeuModelo': pd.Series(ctjr),
    "Correta" : pd.Series(df_temp[coluna_target])
})

print(df_temporario)

df_temporario2 = pd.DataFrame({
            'árvore': pd.Series(dt_predictions),
            'knn': pd.Series(knn_predictions),
            'bayes': pd.Series(nb_predictions)
})
print(df_temporario2)

fim = pd.concat([df_temporario, df_temporario2], axis=1)
print(fim)