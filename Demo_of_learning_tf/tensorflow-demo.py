import tensorflow as tf
import numpy as np

x_data = np.random.rand(100)
y_data = x_data*0.1 + 0.2

#构造线性模型
b = tf.Variable(0.)
k = tf.Variable(0.)
y = k*x_data + b

#代价函数
loss = tf.reduce_mean(tf.square(y_data - y))
#定义一个梯度下降法来进行训练的优化器
optimizer = tf.train.GradientDescentOptimizer(0.01)
#最小化代价函数
train = optimizer.minimize(loss)

#初始化变量
init  = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(201):
        sess.run(train)
        if step%20 == 0:
            print(step, sess.run([k, b]))