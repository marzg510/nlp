import numpy as np
import tensorflow as tf

w = tf.Variable(tf.zeros([2,1]))
b = tf.Variable(tf.zeros([1]))

x = tf.placeholder(tf.float32, shape=[None,2])
t = tf.placeholder(tf.float32, shape=[None,1])
y = tf.nn.sigmoid(tf.matmul(x,w) + b)
cross_entropy = - tf.reduce_sum(t * tf.log(y) + (1 - t) * tf.log(1 - y))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
# 0.1=学習率

# 評価
correct_prediction = tf.equal(tf.to_float(tf.greater(y, 0.5)), t)

X = np.array([[0,0],[0,1],[1,0],[1,1]])
Y = np.array([[0]  ,[1]  ,[1]  ,[1]])
#Y = np.array([[0]  ,[0]  ,[0]  ,[1]]) #AND
#Y = np.array([[0]  ,[1]  ,[1]  ,[0]]) #XOR

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(200):
  sess.run(train_step, feed_dict={
    x: X,
    t: Y
  })
  # free_dictは値を変数に入れること(x<=X,y<=Y)

# 学習結果の確認
classified = correct_prediction.eval(session=sess,feed_dict={
    x: X,
    t: Y
})
print("classified",classified)

prob = y.eval(session=sess,feed_dict={
    x: X,
    t: Y
})
print("prob")
print(prob)

print("w:",sess.run(w))
print("b:",sess.run(b))

