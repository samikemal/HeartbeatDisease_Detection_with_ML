import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Dropout
import keras.optimizers
import pandas as pd
import numpy as np


df = pd.read_csv("mitbih_train.csv",header=0)
df_train = pd.read_csv("mitbih_test.csv",header=0)
veri=df.to_numpy()
a = np.random.permutation(veri)
veri_test = df_train.to_numpy()
a_test =np.random.permutation(veri_test)
giris = a[:, 0:187]
cikis = a[:,187]
x_test = a_test[:, 0:187]
y_test = a_test[: , 187]
test_tuple = (x_test,y_test)

model = Sequential()
model.add(Dense(64,activation='relu',input_dim=187))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5,activation='softmax'))

optimizers= keras.optimizers.Adam(0.01)
loss=keras.losses.sparse_categorical_crossentropy

model.compile(optimizer='adam',loss=loss,metrics=['accuracy'])


history = model.fit(giris,cikis,epochs=50,batch_size=32,validation_data=test_tuple)

# plt.plot(history.history['acc'],color='blue')
# plt.plot(history.history['val_acc'],color='yellow')
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()

# # Plot training & validation loss values
# plt.plot(history.history['loss'],color='blue')
# plt.plot(history.history['val_loss'],color='yellow')
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()