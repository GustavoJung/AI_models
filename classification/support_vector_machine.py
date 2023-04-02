import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report, f1_score
from keras.callbacks import ModelCheckpoint

df = pd.read_csv('../datasets/cell_samples.csv')

#coluna BareNuc tem dados que nao sao ints
df = df[pd.to_numeric(df['BareNuc'], errors='coerce').notnull()]
df['BareNuc'] = df['BareNuc'].astype('int')

X = np.asarray(df.drop(['Class','ID'], axis=1))
Y = np.asarray(df['Class']) #benigno 2 maligno 4

X_train,X_test, Y_train,Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)

kernels = ['poly', 'sigmoid', 'rbf', 'linear']
#checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
for kernel in kernels:
    model = svm.SVC(kernel=kernel, C=1.0, random_state=42)
    model.fit(X_train, Y_train)
    pred = model.predict(X_test)
    report = classification_report(Y_test,pred)
    print(f'Funcao {kernel}')
    print(model.decision_function(X_train[:10]))
    print('---- Report')
    f1 = f1_score(Y_test, pred, average='micro')
    print(f1)
    
