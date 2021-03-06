from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np

df=pd.read_csv("mitbih_train.csv")
veri=df.to_numpy()
x_train=veri[:,0:187]
y_train=veri[:,187]

df_test=pd.read_csv("mitbih_test.csv")
veri_test=df_test.to_numpy()
x_test=veri_test[:,0:187]
y_test=veri_test[:,187]

clf =  GaussianNB()
clf.fit(x_train,y_train)

clf_pf = GaussianNB()
clf_pf.partial_fit(x_train,y_train, np.unique(y_train))

score = clf.score(x_test, y_test)
score2= clf_pf.score(x_test, y_test)
y_predict=clf.predict_proba(x_test)