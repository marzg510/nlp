import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

model = Sequential()

# 入力層~隠れ層
model.add(Dense(input_dim=2, units=2))
model.add(Activation('sigmoid'))

# 隠れ層~出力層
model.add(Dense(units=1))
model.add(Activation('sigmoid'))

# 確率的勾配降下法
model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.1)) # lr=学習率(learning rate)

# XOR gate
X = np.array([[0,0],[0,1],[1,0],[1,1]])
Y = np.array([[0]  ,[1]  ,[1]  ,[0]])

# 学習
model.fit(X, Y, epochs=4000, batch_size=1)

# 分類（発火）確認
classes = model.predict_classes(X, batch_size=1)
prob = model.predict_proba(X, batch_size=2)

print()
print("classified",classes)
print(Y==classes)
print()
print("probability:")
print(prob)
