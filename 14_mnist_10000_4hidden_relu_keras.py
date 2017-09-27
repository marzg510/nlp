import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from sklearn import datasets
from sklearn.model_selection import train_test_split
#from sklearn.utils import shuffle

mnist = datasets.fetch_mldata('MNIST original', data_home='.')

n = len(mnist.data)
N = 10000
indices = np.random.permutation(range(n))[:N]  # ランダムにN枚を選択
X = mnist.data[indices]
y = mnist.target[indices]
Y = np.eye(10)[y.astype(int)] # 1-of-K表現に変換

# 訓練データとテストデータに分ける
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8)

'''
モデル設定
'''
n_in = len(X[0]) # 784 入力層のノード(ニューロン)数
n_hidden = 200    # 隠れ層のノード(ニューロン)数
n_out = len(Y[0]) # 10 出力層のノード(ニューロン)数

model = Sequential()
# 入力層~隠れ層
model.add(Dense(n_hidden, input_dim=n_in))
model.add(Activation('relu'))

model.add(Dense(n_hidden))
model.add(Activation('relu'))

model.add(Dense(n_hidden))
model.add(Activation('relu'))

model.add(Dense(n_hidden))
model.add(Activation('relu'))

# 隠れ層~出力層
model.add(Dense(n_out))
model.add(Activation('softmax'))

# 確率的勾配降下法
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01), metrics=['accuracy']) # lr=学習率(learning rate)


'''
モデル学習
'''
epochs = 1000
batch_size = 100
model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size)

# 予測
loss_and_metrics = model.evaluate(X_test, Y_test)

print()
print(loss_and_metrics)

#Epoch 1000/1000
#8000/8000 [==============================] - 0s - loss: 14.4720 - acc: 0.1021     
#1888/2000 [===========================>..] - ETA: 0s
#[14.602994255065918, 0.094]
#
#real	14m46.777s
#user	33m13.556s
#sys	0m59.644s

