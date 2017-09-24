import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from sklearn import datasets
#from sklearn.model_selection import train_test_split
#from sklearn.utils import shuffle

N = 300
X, y = datasets.make_moons(N, noise=0.3)

Y = y.reshape(N, 1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8)

model = Sequential()

# 入力層~隠れ層
model.add(Dense(3, input_dim=2))
model.add(Activation('sigmoid'))

# 隠れ層~出力層
model.add(Dense(1))
model.add(Activation('sigmoid'))

# 確率的勾配降下法
model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.05), metrics=['accuracy']) # lr=学習率(learning rate)


# 学習
model.fit(X, Y, epochs=500, batch_size=20)

# 予測
loss_and_metrics = model.evaluate(X_test, Y_test)

print()
print(loss_and_metrics)
