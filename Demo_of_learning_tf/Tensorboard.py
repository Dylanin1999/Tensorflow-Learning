import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#载入数据
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)


#每次批次的大小
batch_size = 64

#计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

#命名空间
with tf.name_scope('input'):
    #定义两个placeholder
    x = tf.placeholder(tf.float32, [None, 784], name='x_input')
    y = tf.placeholder(tf.float32, [None, 10], name='y_input')

lr = tf.Variable(1, dtype=tf.float32)

#创建神经网络
#命名空间
with tf.name_scope('layer1'):
    W1 = tf.Variable(tf.truncated_normal([784, 200], stddev=0.1), name="W1")
    b1 = tf.Variable(tf.zeros([1, 200]), name="b1")
    layer1 = tf.nn.relu(tf.matmul(x, W1)+b1)
    layer1_dropout = tf.nn.dropout(layer1, rate=1-0.5)

with tf.name_scope('layer2'):
    W2 = tf.Variable(tf.truncated_normal([200,  100], stddev=0.1), name="W2")
    b2 = tf.Variable(tf.zeros([1, 100]), name="b2")
    layer2 = tf.nn.relu(tf.matmul(layer1_dropout, W2)+b2)
    layer2_dropout = tf.nn.dropout(layer2, rate=1-0.5)


    W3 = tf.Variable(tf.truncated_normal([100, 50], stddev=0.1))
    b3 = tf.Variable(tf.zeros([1, 50]))
    layer3 = tf.nn.relu(tf.matmul(layer2_dropout, W3)+b3)
    layer3_dropout = tf.nn.dropout(layer3, rate=1-0.5)

    W4 = tf.Variable(tf.truncated_normal([50,  10], stddev=0.1))
    b4 = tf.Variable(tf.zeros([1, 10]))
    prediction = tf.nn.softmax(tf.matmul(layer3_dropout, W4)+b4)


#loss func
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=prediction))
train_step = tf.train.AdadeltaOptimizer(lr).minimize(loss)

#初始化变量
init = tf.global_variables_initializer()

#初始化变量
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))

#求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    writer=tf.summary.FileWriter('logs/', sess.graph)
    for epoch in range(2):
        sess.run(tf.assign(lr, 1 * (0.98 ** epoch)))
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
        learning_rate = sess.run(lr)
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Iter " + str(epoch) + ",Testing Accuracy" + str(acc))
