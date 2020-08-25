from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd

df=pd.read_csv("mitbih_train.csv")
veri=df.to_numpy()
x_train=veri[:,0:187]
y_train=veri[:,187]

df_test=pd.read_csv("mitbih_test.csv")
veri_test=df_test.to_numpy()
x_test=veri_test[:,0:187]
y_test=veri_test[:,187]

clf = RandomForestClassifier(max_depth=20, n_estimators=10, max_features=5)
clf.fit(x_train,y_train)
score = clf.score(x_test, y_test)
y_predict=clf.predict(x_test)

conf=confusion_matrix
confMatrix=conf(y_test, y_predict)
