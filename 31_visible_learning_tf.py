import numpy as np
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

def inference(x, keep_prob, n_in, n_hiddens, n_out):
    # モデルの定義
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.zeros(shape)
        return tf.Variable(initial)

    # 入力層ー隠れ層、隠れ層ー入力層
    for i, n_hidden in enumerate(n_hiddens):
        if i == 0:
            input = x
            input_dim = n_in
        else:
            input = output
            input_dim = n_hiddens[i-1]

        W = weight_variable([input_dim, n_hidden])
        b = bias_variable([n_hidden])

        h = tf.nn.relu(tf.matmul(input, W) + b)
        output = tf.nn.dropout(h, keep_prob)

    # 隠れ層ー出力層
    W_out = weight_variable([n_hiddens[-1], n_out])
    b_out = bias_variable([n_out])
    y = tf.nn.softmax(tf.matmul(output, W_out) + b_out)
    return y

def loss(y, t):
    # 誤差関数の定義
    cross_entropy = tf.reduce_mean( -tf.reduce_sum(t * tf.log(y), reduction_indices=[1]))
    return cross_entropy

def training(loss):
    # 学習アルゴリズムの定義
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train_step = optimizer.minimize(loss)
    return train_step

# main
if __name__ == '__main__':
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
    # 2.モデル設定
    n_in = len(X[0]) # 784 入力層のノード(ニューロン)数
    n_hiddens = [200,200,200]    # 隠れ層のノード(ニューロン)数
    n_out = len(Y[0]) # 10 出力層のノード(ニューロン)数
    x = tf.placeholder(tf.float32, shape=[None,n_in])
    t = tf.placeholder(tf.float32, shape=[None,n_out])
    keep_prob = tf.placeholder(tf.float32)
    y = inference(x, keep_prob, n_in=n_in, n_hiddens=n_hiddens, n_out=n_out)
    # 誤差関数
    loss = loss(y, t)
    # 評価
    correct_prediction = tf.equal(tf.to_float(tf.greater(y, 0.5)), t)
    # 予測精度の評価
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 3.モデル学習
    train_step = training(loss)
    epochs = 50
    batch_size = 200
    n_batches = N // batch_size
    p_keep = 0.5
    history = {
        'val_loss': [],
        'val_acc': []
    }

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        X_, Y_ = shuffle(X_train, Y_train)
        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            sess.run(train_step, feed_dict={
              x: X_[start:end],
              t: Y_[start:end],
              keep_prob: p_keep
            })
        # 4.モデル評価
        # 検証データを用いた評価
        val_loss = loss.eval(session=sess, feed_dict={
            x: X_validation,
            t: Y_validation,
            keep_prob: 1.0
        })
        val_acc = accuracy.eval(session=sess, feed_dict={
            x: X_validation,
            t: Y_validation,
            keep_prob: 1.0
        })
        # 検証データに対する学習の進み具合を記録
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        if epoch % 10 == 0:
            print('epoch:',epoch)

    # 4.モデル評価
    accuracy_rate = accuracy.eval(session=sess, feed_dict={
        x: X_test,
        t: Y_test,
        keep_prob: 1.0
    })
    print('accuracy: ', accuracy_rate)
    # 可視化
    plt.rc('font', family='serif')
    fig = plt.figure()
    ax_acc = fig.add_subplot(111)
    ax_acc.plot(range(epochs), history['val_acc'], label='acc', color='black')
    ax_loss = ax_acc.twinx()
    ax_loss.plot(range(epochs), history['val_loss'], label='loss', color='gray')
    plt.xlabel('epochs')
#    plt.ylabel('validation loss')
    plt.savefig('mnist_relu_tf.eps')
    plt.show()

