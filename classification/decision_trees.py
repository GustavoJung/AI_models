import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.tree as tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics

## PREPARAÇÃO DE DADOS

df = pd.read_csv('../datasets/drug200.csv')
#infos do tamanho do DataFrame = df.shape

#X será correspondente aos dados usados para fazer a previsao
#Y será correspondente a variável 'resultado'

X = df[['Age', 'Sex', 'BP', 'Cholesterol','Na_to_K']].values

#Ajustando as variáveis qualitativas para novas colunas
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F', 'M']) #F = 0 e M = 1
X[:,1] = le_sex.transform(X[:,1])

le_bp = preprocessing.LabelEncoder()
le_bp.fit(['LOW', 'NORMAL', 'HIGH'])  #Low = 0, Normal = 1, High = 2
X[:,2] = le_bp.transform(X[:,2])

le_chol = preprocessing.LabelEncoder()
le_chol.fit(['NORMAL', 'HIGH']) #Normal = 0, High = 1
X[:,3] = le_chol.transform(X[:,3])

Y = df[['Drug']]


## Modelando a Árvore de decisão

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=3)


#entroyp = Medida da quantidade de aleatoriedade ou incerteza nos dados. Exemplo:
# 1 Droga A                 3 Droga A
# 7 Droga B                 5 Droga B
# Baixa entropia            Alta entropia

#Quanto menos entropia, mais puro o nó da arvore

#criando o modelo
arvoreDecisao = DecisionTreeClassifier(criterion="entropy", max_depth=4)
#Treinando o modelo com os grupos de treino
arvoreDecisao.fit(X_train, Y_train)
#fazendo previsoes com o grupo de teste
arvoreValoresPreditos = arvoreDecisao.predict(X_test)
#Conferindo se as previsoes batem com os resultados 

#print(arvoreValoresPreditos[0:5])
#print(Y_test[0:5])

#Avaliando o modelo
print('Precisao do modelo: ', metrics.accuracy_score(Y_test, arvoreValoresPreditos))

#Visualizando a árvore
tree.plot_tree(arvoreDecisao)
plt.show()
