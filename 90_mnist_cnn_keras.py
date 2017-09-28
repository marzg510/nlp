import os
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adadelta
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.layers.core import Dropout
from keras.initializers import TruncatedNormal
from keras.layers import Flatten, Conv2D, MaxPooling2D
#from keras.datasets import mnist

# 1.データの準備
(X, Y), (X_dummy, Y_dummy) = keras.datasets.mnist.load_data()
n = X.shape[0]
img_rows = X.shape[1]
img_cols = X.shape[2]
input_shape = (img_rows, img_cols, 1)
X = X.reshape(n, img_rows, img_cols, 1)
N = 30000  # MNISTの一部を使う
N_train = 20000
N_validation = 4000
indices = np.random.permutation(range(n))[:N]  # ランダムにN枚を選択
X = X[indices]
Y = Y[indices]
# Xをfloatに変換して正規化
X = X.astype('float32')
X /= 255
# Yはint32に変換して1-of-K型にする
Y = Y.astype('int32')
Y = keras.utils.np_utils.to_categorical(Y, 10)
# 訓練データとテストデータに分ける
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=N_train)
# 訓練データをさらに訓練データと検証データに分割
X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=N_validation)

###
# 2.モデル設定
###
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3),
                 activation='relu',
                 input_shape=input_shape
                 )
        )
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# 確率的勾配降下法
model.compile(loss='categorical_crossentropy', optimizer=Adadelta(), metrics=['accuracy'])

###
# 3.モデル学習
###
epochs = 12
batch_size = 120
hist = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,
                validation_data=(X_validation ,Y_validation))

# 予測
loss_and_metrics = model.evaluate(X_test, Y_test)
print()
print(loss_and_metrics)
# 可視化
val_loss = hist.history['val_loss']
val_acc = hist.history['val_acc']
print(val_loss)
print(val_acc)

plt.rc('font', family='serif')
fig = plt.figure()
# 精度
ax_acc = fig.add_subplot(111)
ax_acc.plot(range(epochs), val_acc, label='acc', color='black')
# 損失
ax_loss = ax_acc.twinx()
ax_loss.plot(range(epochs), val_loss, label='loss', color='gray')
plt.xlabel('epochs')
file,ext = os.path.splitext(os.path.basename(__file__))
plt.savefig(file+'.eps')

plt.show()

# 学習のみ 50epoch
#Epoch 50/50
#16000/16000 [==============================] - 115s - loss: 0.0153 - acc: 0.9951 - val_loss: 0.0534 - val_acc: 0.9855
# 9984/10000 [============================>.] - ETA: 0s 
#real	85m45.373s
#user	261m18.604s
#sys	24m29.952s

# 学習＆評価
#Epoch 12/12
#16000/16000 [==============================] - 75s - loss: 0.0508 - acc: 0.9837 - val_loss: 0.0587 - val_acc: 0.9858
# 9952/10000 [============================>.] - ETA: 0s 
#
#real	14m20.398s
#user	42m27.916s
#sys	3m59.964s

