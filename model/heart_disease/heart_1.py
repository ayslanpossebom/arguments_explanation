#Geração da lista de argumentos

import pandas as pd
from itertools import combinations
from collections import Counter
import os

url = os.path.dirname(os.path.abspath(__file__)) + "\\"



#CONFIGURAÇÃO

argumentos_heart = url + 'heart_argumentos_todos.ob'


dataset_original = url + "heart_disease.csv" 
dataset_binarizado = url + "heart_disease_binarizado.csv"
dataset_binarizado_amostras = url + "heart_disease_binarizado_amostras_todos.csv"
coluna_target = "num"






###################################################################################
class Argumento:
    def __init__(self, df_original="", coluna_target="", total_amostras_por_classe=0):
        self.argumentos = []
        argumentos_invalidos = []
        argumentos_validos = []
        print("Validando argumentos...")

        if(total_amostras_por_classe != 0):
            # Embaralhando o DataFrame
            df_original = df_original.sample(frac=1).reset_index(drop=True)
            # Depois de embaralhar, selecione as amostras para cada classe possivel
            df = df_original.groupby(coluna_target).head(total_amostras_por_classe)
        else:
            df = df_original
        #df.to_csv(url + "heart_disease_binarizado_amostras.csv", index=False)
        #exit()

        atual = 0
        total = len(df)
        #Para cada argumento a ser analisado
        for index, row in df.iterrows():
            if(atual % 10 == 0):
                print(f"\t{atual} de {total}")
            atual += 1

            #PASSO 1: Obter os atributos que estão com 1
            atributos_com_1 = [col for col in df.columns if col != coluna_target and row[col] == 1]
            #print("COLUNA TARGET: ", row[coluna_target])
            

            #PASSO 2: Gerar todas as combinações possíveis dos argumentos com 1
            xxx = 1
            for i in range(1, len(atributos_com_1)):
                

                combinacoes = combinations(atributos_com_1, i)

                #Para cada combinação possível
                for combinacao in combinacoes:

                    temp = set(combinacao)

                    #se quiser saber se apenas existe ou não algum subconjunto
                    #subconjunto_valido = any(conjunto.issubset(temp) for conjunto in argumentos_validos)
                    #subconjunto_invalido = any(conjunto.issubset(temp) for conjunto in argumentos_invalidos)

                    #se quiser obter todos os subconjuntos conjuntos
                    #subconjunto_valido = [conjunto for conjunto in argumentos_validos if conjunto.issubset(temp)]
                    #subconjunto_invalido = [conjunto for conjunto in argumentos_invalidos if conjunto.issubset(temp)]

                    #para obter as interseções

                    if(temp not in argumentos_validos and temp not in argumentos_invalidos):
                        conjuntos_com_intersecao = [conjunto for conjunto in argumentos_validos if conjunto.issubset(temp)]

                    
                        if(len(conjuntos_com_intersecao) == 0):
                            condicao = (df[list(temp)] == True).all(axis=1)
                            df_filtrado = df.loc[condicao]
                            valores_unicos = df_filtrado[coluna_target].unique()
                            if(len(valores_unicos) == 1):
                                arg = {"premissas": temp, "conclusao": row[coluna_target]}
                                self.argumentos.append(arg)
                                argumentos_validos.append(temp)         
                                print("ADD ", arg)

                            else:
                                argumentos_invalidos.append(temp)
                                #print("Argumento invalido: ", temp, " = ", valores_unicos)
                        #else:
                            #print("Subargumento ja foi aceito:", temp, " = ", conjuntos_com_intersecao)
                    #else:
                        #print("argumento repetido", temp)

                                        
                    xxx += 1


        print("Total de argumentos válidos: ", len(argumentos_validos))
        print("Total de argumentos inválidos: ", len(argumentos_invalidos))





##############################################################

#Carregar dataset
df = pd.read_csv(dataset_binarizado_amostras)
resultados_possiveis = df[coluna_target].unique()


#EXECUTAR UMA VEZ PARA CRIAR A LISTA DE ARGUMENTOS
def computarArgumentos(df, coluna_target, quantia_para_teste=0):
    print(len(df))

    #Criar objeto com a lista de argumentos válidos
    argumentos = Argumento(df_original=df, coluna_target=coluna_target, total_amostras_por_classe=quantia_para_teste)
    print("Total de argumentos válidos: ", len(argumentos.argumentos))

    import pickle
    with open(argumentos_heart, 'wb') as fp:
        pickle.dump(argumentos.argumentos, fp)
    return argumentos

computarArgumentos(df, coluna_target)
exit()







#EXPERIMENTOS:
#DATASET AMOSTRA COM 15 EXEMPLARES DE CADA UM GEROU 1944 DE ARGUMENTOS VALIDOS e 78416 INVALIDOS
#MACHINE LEARNING TREINO COM AMOSTRAS E TESTE COM TODOS: acurácia KNN=0.492, Tree=0,492, bayes=0.516, SVM=0.623




#PARA TODOS OS VALORES:
#Total de argumentos válidos:  4916
#Total de argumentos inválidos:  237285
#Aproximadamente 3:30 horas