Tensorflow AND gate
======================


## references
http://smzr.blog.jp/archives/1058068647.html


## run
```
import tensorflow as tf

x_data=[[0,0],[0,1],[1,0],[1,1]]
y_data=[[0,1],[0,1],[0,1],[1,0]]  # [0,1]=0,[1,0]=1

# variables
x = tf.placeholder("float",[None,2])
W = tf.Variable(tf.zeros([2,2]))
b = tf.Variable(tf.zeros([2]))
y_ = tf.placeholder("float",[None,2])

# 出力層
y=tf.nn.softmax(tf.matmul(x,W)+b) 

# 損失関数(交差エントロピー)
cross_entropy=-tf.reduce_sum(y_*tf.log(y))

# 学習も勾配降下法。学習係数は何も考えずに0.5。
train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# xとy_がplaceholderなのでfeed_dictで値を渡す。
feed_dict={x:x_data,y_:y_data}


sess=tf.Session()
sess.run(tf.global_variables_initializer())

# 1000 epoch
for step in range(1001):
  sess.run(train_step,feed_dict=feed_dict)
  #100回ごとにテスト結果を表示。
  if step%100==0:
    print("step:",step)
    # テスト用データをx_inputに入れて後でfeed_dictで指定。
    for x_input in [[0,0],[0,1],[1,0],[1,1]]:
      print(x_input, sess.run(y, feed_dict={x:[x_input]}))

```

