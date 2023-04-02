import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

df = pd.read_csv('../datasets/ChurnData.csv')

X = np.asarray(df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'callcard', 'wireless']])
Y = np.asarray(df['churn'])


#print(df.shape)
#print(df.columns)

scaler = StandardScaler()
X = scaler.fit(X).transform(X.astype(float))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


#numerical optimizers to find parameters, including ‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’
#C parameter indicates inverse of regularization strength which must be a positive float. Smaller values specify stronger regularization.
solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
for solver in solvers:
    logisticRegression = LogisticRegression(C=0.01, solver=solver)
    logisticRegression.fit(X_train, Y_train)

    pred = logisticRegression.predict(X_test)

    #**predict_proba**  returns estimates for all classes, ordered by the label of classes.
    #  So, the first column is the probability of class 0, P(Y=0|X), and second column is probability of class 1, P(Y=1|X):
    pred_probabilidade = logisticRegression.predict_proba(X_test)
    
    class_report = classification_report(Y_test, pred)
    print(f'Solver {solver}: Stats')
    print(class_report)
