import numpy as np
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

rng = np.random.RandomState(123)

M = 2  # dimension
K = 3  # num of class
n = 100 # num of data per class
N = n * K # num of data

X1 = rng.randn(n,M) + np.array( [0,10])
X2 = rng.randn(n,M) + np.array([ 5, 5])
X3 = rng.randn(n,M) + np.array([10, 0])
Y1 = np.array([[1, 0, 0] for i in range(n)])
Y2 = np.array([[0, 1, 0] for i in range(n)])
Y3 = np.array([[0, 0, 1] for i in range(n)])

X = np.concatenate((X1,X2,X3), axis=0)
Y = np.concatenate((Y1,Y2,Y3), axis=0)

model = Sequential()
model.add(Dense(input_dim=M, units=K))
model.add(Activation('softmax'))

# 確率的勾配降下法
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1))

# 学習
minibatch_size=1
model.fit(X, Y, epochs=200, batch_size=minibatch_size)

# 分類（発火）確認
X_, Y_ = shuffle(X,Y)
classes = model.predict_classes(X_[0:10], batch_size=minibatch_size)
prob = model.predict_proba(X_[0:10], batch_size=minibatch_size)

print('classified:')
print(np.argmax(model.predict(X[0:10]),axis=1) == classes)
print()
print('output probability:')
print(prob)

#print('w')
#print(sess.run(W))
#print('b')
#print(sess.run(b))

