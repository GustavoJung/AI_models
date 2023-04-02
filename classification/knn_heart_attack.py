import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


df = pd.read_csv('../datasets/heart/heart.csv')
df.drop_duplicates(keep='first', inplace=True)
#colunas: age  sex  cp  trtbps  chol  fbs  restecg  thalachh  exng  oldpeak  slp  caa  thall  output

X = df.drop('output', axis=1)
Y = df['output']


scaler = StandardScaler()
X = scaler.fit(X).transform(X.astype(float))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#melhores parametros
knn_params = {'n_neighbors':[3,5,7],
              'leaf_size':[10,20,30],
              'p':[1,2],
              'weights':['uniform', 'distance']
              }

knn_model_test = KNeighborsClassifier()
search_params = GridSearchCV(knn_model_test, knn_params, cv=5)
search_params.fit(X_train, Y_train)
best_params = search_params.best_params_


#rede real
knn_model = KNeighborsClassifier(**best_params)
knn_model.fit(X_train,Y_train)

previsoes = knn_model.predict(X_test)

c_report = classification_report(Y_test,previsoes)
print(c_report)
print(previsoes[0:5])
print(Y_test[0:5].values)


#Regressao Logística
#Bom para prever valores binários
