import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

model = Sequential([
    Dense(input_dim=2, units=1), # w1x1 + w2x2 + b
    Activation('sigmoid')
    # y=σ(w1x1 + w2x2 +b)
])

# 確率的勾配降下法
model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.1)) # lr=学習率(learning rate)

# OR gate
X = np.array([[0,0],[0,1],[1,0],[1,1]])
Y = np.array([[0]  ,[1]  ,[1]  ,[1]])

# 学習
model.fit(X, Y, epochs=200, batch_size=1)

# 分類（発火）確認
classes = model.predict_classes(X, batch_size=1)
prob = model.predict_proba(X, batch_size=1)

print()
print("classified",classes)
print(Y==classes)
print()
print("prob",prob)
