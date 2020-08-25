from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

df=pd.read_csv("mitbih_train.csv")
veri=df.to_numpy()
x_train=veri[:,0:187]
y_train=veri[:,187]

df_test=pd.read_csv("mitbih_test.csv")
veri_test=df_test.to_numpy()
x_test=veri_test[:,0:187]
y_test=veri_test[:,187]

clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(x_train,y_train)
score = clf.score(x_test, y_test)
y_predict=clf.predict_proba(x_test)

#KNN makalede belirtildiği üzere en uzun çalışma zamanına sahip algoritmamız çünkü 
#bütün veriler için teker teker tüm verileri tekrar değerlendiriyor
#ancak görüldüğü üzere çok yüksek bir orana sahip