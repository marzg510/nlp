import numpy as np
import tensorflow as tf

X = np.array([[0,0],[0,1],[1,0],[1,1]])
Y = np.array([[0]  ,[1]  ,[1]  ,[0]])

x = tf.placeholder(tf.float32, shape=[None,2])
t = tf.placeholder(tf.float32, shape=[None,1])

# 入力層〜隠れ層
w = tf.Variable(tf.truncated_normal([2,2]))
b = tf.Variable(tf.zeros([1]))
h = tf.nn.sigmoid(tf.matmul(x,w) + b)

# 隠れ層〜出力層
v = tf.Variable(tf.truncated_normal([2,1]))
c = tf.Variable(tf.zeros([1]))
y = tf.nn.sigmoid(tf.matmul(h,v) + c)

cross_entropy = - tf.reduce_sum(t * tf.log(y) + (1 - t) * tf.log(1 - y))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
# 0.1=学習率

# 評価
correct_prediction = tf.equal(tf.to_float(tf.greater(y, 0.5)), t)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(4000):
  sess.run(train_step, feed_dict={
    x: X,
    t: Y
  })
  # feed_dictは値を変数に入れること(x<=X,t<=Y)

  if epoch % 1000 == 0:
      print('epoch:',epoch)

# 学習結果の確認
classified = correct_prediction.eval(session=sess,feed_dict={
    x: X,
    t: Y
})

prob = y.eval(session=sess,feed_dict={
    x: X
})
print("classified")
print(classified)
print()
print("prob")
print(prob)

