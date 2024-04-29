#Geração da lista de argumentos

import pandas as pd
from itertools import combinations
from collections import Counter
import os

url = os.path.dirname(os.path.abspath(__file__)) + "\\"



#CONFIGURAÇÃO

dataset_original = url + "seeds.csv" 
dataset_binarizado = url + "seeds_binarizado.csv"
dataset_binarizado_amostras = url + "seeds_binarizado_amostras.csv"

coluna_target = "Type"

argumentos_seeds = url + 'seeds_argumentos.ob'





###################################################################################
class Argumento:
    def __init__(self, df_original="", coluna_target="", total_amostras_por_classe=0):
        self.argumentos = []
        argumentos_invalidos = []
        argumentos_validos = []

        self.total_validos = 0
        self.total_invalidos = 0
        self.total_essenciais = 0
        self.total_argumentos_geral = 0        

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
                    self.total_argumentos_geral += 1
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
                            self.total_validos += 1
                            if(len(valores_unicos) == 1):
                                arg = {"premissas": temp, "conclusao": row[coluna_target]}
                                self.argumentos.append(arg)
                                argumentos_validos.append(temp)         
                                print("ADD ", arg)
                                self.total_essenciais += 1
                            else:
                                argumentos_invalidos.append(temp)
                                self.total_invalidos += 1
                                #print("Argumento invalido: ", temp, " = ", valores_unicos)
                        #else:
                            #print("Subargumento ja foi aceito:", temp, " = ", conjuntos_com_intersecao)
                    #else:
                        #print("argumento repetido", temp)

                                        
                    xxx += 1


        print("Total de argumentos válidos: ", len(argumentos_validos))

        for i in range(3):
            print(i, argumentos_validos[i])
            
        print("Total de argumentos inválidos: ", len(argumentos_invalidos))
        for i in range(3):
            print(i, argumentos_invalidos[i])

        





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

    print("DADOS:")
    print("Total de argumentos: ", argumentos.total_argumentos_geral)
    print("Total de argumentos válidos: ", argumentos.total_validos)
    print("Total de argumentos inválidos: ", argumentos.total_invalidos)
    print("Total de argumentos essenciais: ", argumentos.total_essenciais)


    import pickle
    with open(argumentos_seeds, 'wb') as fp:
        pickle.dump(argumentos.argumentos, fp)
    return argumentos

computarArgumentos(df, coluna_target)
exit()
