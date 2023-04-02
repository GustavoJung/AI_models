import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

colunas = ['fLength','fWidth', 'fSize', 'fConc', 'fConl','fAsym', 'fM3Loing','fM3Trans','fAlpha','fDist', 'class']
df = pd.read_csv('magic04.data', names=colunas)

df['class'] = (df['class'] == 'g').astype(int)

#mostrar os dados 
"""    for label in colunas[:-1]:
        plt.hist(df[df['class']==1][label], color='blue', label='gamma', alpha=0.7, density=True)
        plt.hist(df[df['class']==0][label], color='red', label='hadron', alpha=0.7, density=True)
        plt.title(label)
        plt.ylabel("Probabilidade")
        plt.xlabel(label)
        plt.legend()
        plt.show()
"""

#datasets com numpy
#train, valid, test = np.split(df.sample(frac=1), [int(0.6*len(df))])
X = df.loc[:, colunas[:-1]]
Y = df['class']


#scalando os dados

scaler = StandardScaler()
scaler.fit(X).transform(X.astype(float))


#como existe mta diferenca entre a quantidade de cada dado, incrementar a quantidade de dados do menor para q fiquem parecidas
ros = RandomOverSampler()


X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size=0.2, random_state=42)
X_train, Y_train = ros.fit_resample(X_train, Y_train)

#Classificando usando KNN
#Encontrando melhores parametros
param_grid = {'n_neighbors':[3,5,7],
              'leaf_size':[10,20,30],
              'p':[1,2],
                'weights':['uniform', 'distance']}

#‘uniform’ : uniform weights. All points in each neighborhood are weighted equally.
#‘distance’ : weight points by the inverse of their distance. in this case, closer neighbors of a query point will have a greater influence than neighbors which are further away.
#algorithm{‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}

knn_model = KNeighborsClassifier()
grid_search = GridSearchCV(knn_model, param_grid, cv=5) #cv é cross validation
grid_search.fit(X_train, Y_train)
best_params = grid_search.best_params_

#Modelo usando os melhores paramettros encontrados
knn = KNeighborsClassifier(**best_params)
knn.fit(X_train,Y_train)

#previsoes
previsoes = grid_search.predict(X_test)

print(classification_report(Y_test, previsoes))



##Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, Y_train)

nb_pred =nb_model.predict(X_test)

#print(classification_report(Y_test, nb_pred))