import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adadelta
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.layers.core import Dropout
from keras.initializers import TruncatedNormal
from keras.callbacks import EarlyStopping

# 1.データの準備
mnist = datasets.fetch_mldata('MNIST original', data_home='.')
n = len(mnist.data)
N = 30000  # MNISTの一部を使う
N_train = 20000
N_validation = 4000
indices = np.random.permutation(range(n))[:N]  # ランダムにN枚を選択
X = mnist.data[indices]
y = mnist.target[indices]
Y = np.eye(10)[y.astype(int)] # 1-of-K表現に変換
# 訓練データとテストデータに分ける
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=N_train)
# 訓練データとをさらに訓練データと検証データに分割
X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=N_validation)

###
# 2.モデル設定
###
n_in = len(X[0]) # 784 入力層のノード(ニューロン)数
n_hiddens = [200,200,200]    # 隠れ層のノード(ニューロン)数
n_out = len(Y[0]) # 10 出力層のノード(ニューロン)数
p_keep = 0.5
activation = 'relu'

model = Sequential()
for i, input_dim in enumerate(([n_in] + n_hiddens)[:-1]):
    model.add(Dense(n_hiddens[i], input_dim=input_dim,kernel_initializer=TruncatedNormal(stddev=0.01)))
    model.add(Activation(activation))
    model.add(Dropout(p_keep))

model.add(Dense(n_out, kernel_initializer=TruncatedNormal(stddev=0.01)))
model.add(Activation('softmax'))

# 確率的勾配降下法
model.compile(loss='categorical_crossentropy', 
            optimizer=Adadelta(rho=0.95),
            metrics=['accuracy']) 

# Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

###
# 3.モデル学習
###
epochs = 300
batch_size = 200
hist = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,
                validation_data=(X_validation ,Y_validation),
                callbacks=[early_stopping])


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
ax_acc.plot(range(len(val_acc)), val_acc, label='acc', color='black')
# 損失
ax_loss = ax_acc.twinx()
ax_loss.plot(range(len(val_loss)), val_loss, label='loss', color='gray')
plt.xlabel('epochs')
#    plt.ylabel('validation loss')
file,ext = os.path.splitext(os.path.basename(__file__))
plt.savefig(file+'.eps')
plt.show()
# 30回前後で打ち切り
#[0.16939273901265114, 0.96179999999999999]

