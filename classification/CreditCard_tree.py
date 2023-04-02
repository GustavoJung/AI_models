from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
import time
import warnings
from snapml import DecisionTreeClassifier
warnings.filterwarnings('ignore')

#lendo os dados
df = pd.read_csv('../datasets/creditcard.csv')
#print('Linhas: ' + str(len(df)))
#print('Colunas: ' + str(len(df.columns)))

#Incrementando o número de dados
n_replicacoes = 10

df_incrementado = pd.DataFrame(np.repeat(df.values, n_replicacoes, axis=0), columns=df.columns)
# Cada linha representa uma transação, cada uma tem 31 variaveis. A ultima coluna é chamada Class (1 é fraude, 0 não é) e é a q deve ser prevista

#### As columas chamadas Vn foram alteradas para não expor os dados

#Set das classes distinstas (mostra que são 0 e 1)
labels = df_incrementado.Class.unique()
#print(labels)

#conta quantas vezes cada uma aparece
sizes = df_incrementado.Class.value_counts().values
#print(sizes)

#plotando num gráfico
#fig, ax = plt.subplots()
#ax.pie(sizes, labels=labels, autopct='%1.3f%%')
#ax.set_title('Qtdade de variáveis a serem previstas')
#plt.show()
#plt.hist(df_incrementado.Amount.values, histtype='bar', facecolor='g')

#print('Valor minimo: ', np.min(df_incrementado.Amount.values))
#print('Valor maximo: ', np.max(df_incrementado.Amount.values))


## Preparando dados do dataset
df_incrementado.iloc[:, 1:30] = StandardScaler().fit_transform(df_incrementado.iloc[:, 1:30])
df_processado = df_incrementado.values

#X sera o grupo de variaveis que representam os atributos independetes usados para prever valores
# A coluna do tempo será removida (a coluna 0)
X = df_processado[:, 1:30]

# Y variavel dependente
Y = df_processado[:, 30]

#Normalização
X = normalize(X, norm="l1")



#Sets de treino e teste
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)



#modelo de arvore de decisão usando Scikit-Learn

#armazena os pesos da amostra para ser usado como input na rotina de treino, assim
#o modelo leva em conta o desequilibrio de classe nos dados

w_train = compute_sample_weight('balanced', Y_train)

sklearn_dt = DecisionTreeClassifier(max_depth=4, random_state=35)

#treinando o modelo
t0 = time.time()
sklearn_dt.fit(X_train, Y_train, sample_weight=w_train)
sklearn_time = time.time() - t0
print('[Scikit Learn] Training time (s):  {0:.5f}'.format(sklearn_time))


#modelo de arvore de decisao usando Snap ML

#caso ainda nao tenha, computador os pesos (w_train acima)

#Snapml permite o uso de GPU para treinar
#para usar, aplicar o parametro use_gpu=True

snapml_dt = DecisionTreeClassifier(max_depth=4, random_state=45, n_jobs=4)

#Treinando o modelo
t0 = time.time()
snapml_dt.fit(X_train, Y_train, sample_weight=w_train)
snapml_dt_time = time.time() - t0
print('[SnapML] Training time (s):  {0:.5f}'.format(snapml_dt_time))


##Avaliando os modelos
velocidade_treino = sklearn_time/snapml_dt_time
print('[Arvóre de Decisão] SnapML vs Scikit-Learn speedup: {0:.2f}x '.format(velocidade_treino))


#probabilidade de um exemplo pertencer a classe de fraudes
sklearn_pred = sklearn_dt.predict_proba(X_test)[:,1]

#Pontuação ROC-AUC
sklearn_roc_auc = roc_auc_score(Y_test, sklearn_pred)
print('[Scikit-Learn] ROC-AUC score:  {0:.3f}'.format(sklearn_roc_auc))


#probabilidade de um exemplo pertencer a classe de fraudes
snapml_pred = snapml_dt.predict_proba(X_test)[:,1]

#Pontuação ROC-AUC
snapml_roc_auc = roc_auc_score(Y_test, snapml_pred)
print('[Snapml] ROC-AUC score:  {0:.3f}'.format(snapml_roc_auc))
