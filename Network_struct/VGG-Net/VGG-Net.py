import tensorflow as tf


def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())


def Weights(shape, name="Weight"):
    weight = tf.Variable(tf.truncated_normal(shape, dtype=tf.float32, stddev=1e-1), name=name)
    return weight


def Biases(shape, name='Bias'):
    Bias = tf.Variable(tf.constant(0.1, shape=shape), name=name)
    return Bias


def VGGNet(train_x, train_y, test_x, test_y):

    X = tf.placeholder(tf.float32, [None, 224 * 224 * 3])
    Y = tf.placeholder(tf.float32, [None, 20])
    keep_drop = tf.placeholder(tf.float32)
    batch_num = len(train_x)

    image = tf.reshape(X, [-1, 224, 224, 3])

    #Conv1  3x3x64
    with tf.name_scope('Conv1') as scope:
        W_conv1 = Weights([3, 3, 3, 64])
        b_conv1 = Biases([64])
        Conv1 = tf.nn.conv2d(image, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
        actv1 = tf.nn.relu(Conv1 + b_conv1)

    print_activations(Conv1)
    print_activations(actv1)

    #Conv2  3x3x64
    with tf.name_scope('Conv2') as scope:
        W_conv2 = Weights([3, 3, 64, 64])
        b_conv2 = Biases([64])
        Conv2 = tf.nn.conv2d(actv1, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
        actv2 = tf.nn.relu(Conv2 + b_conv2)
    print_activations(Conv2)
    print_activations(actv2)

    # pooling1  2x2
    with tf.name_scope('max_pooling1') as scope:
        pool1 = tf.nn.max_pool(actv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    print_activations(pool1)

    # Conv3  3x3x128
    with tf.name_scope('Conv3') as scope:
        W_conv3 = Weights([3, 3, 112, 128])
        b_conv3 = Biases([64])
        Conv3 = tf.nn.conv2d(pool1, W_conv3, strides=[1, 1, 1, 1], padding='SAME')
        actv3 = tf.nn.relu(Conv3 + b_conv3)

    print_activations(Conv3)
    print_activations(actv3)

    # Conv4  3x3x128
    with tf.name_scope('Conv4') as scope:
        W_conv4 = Weights([3, 3, 128, 128])
        b_conv4 = Biases([64])
        Conv4 = tf.nn.conv2d(actv3, W_conv4, strides=[1, 1, 1, 1], padding='SAME')
        actv4 = tf.nn.relu(Conv4 + b_conv4)

    print_activations(Conv4)
    print_activations(actv4)

    # pooling2  2x2
    with tf.name_scope('max_pooling2') as scope:
        pool2 = tf.nn.max_pool(actv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    print_activations(pool2)

    # Conv5  3x3x256
    with tf.name_scope('Conv5') as scope:
        W_conv5 = Weights([3, 3, 128, 256])
        b_conv5 = Biases([256])
        Conv5 = tf.nn.conv2d(pool2, W_conv5, strides=[1, 1, 1, 1], padding='SAME')
        actv5 = tf.nn.relu(Conv5 + b_conv5)

    print_activations(Conv5)
    print_activations(actv5)

    # Conv6  3x3x256
    with tf.name_scope('Conv6') as scope:
        W_conv6 = Weights([3, 3, 256, 256])
        b_conv6 = Biases([256])
        Conv6 = tf.nn.conv2d(actv5, W_conv6, strides=[1, 1, 1, 1], padding='SAME')
        actv6 = tf.nn.relu(Conv6 + b_conv6)

    print_activations(Conv6)
    print_activations(actv6)

    # Conv7  3x3x256
    with tf.name_scope('Conv7') as scope:
        W_conv7 = Weights([3, 3, 256, 256])
        b_conv7 = Biases([256])
        Conv7 = tf.nn.conv2d(actv6, W_conv7, strides=[1, 1, 1, 1], padding='SAME')
        actv7 = tf.nn.relu(Conv7 + b_conv7)

    print_activations(Conv7)
    print_activations(actv7)

    # pooling3  2x2
    with tf.name_scope('max_pooling3') as scope:
        pool3 = tf.nn.max_pool(actv7, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    print_activations(pool3)

    # Conv8  3x3x512
    with tf.name_scope('Conv8') as scope:
        W_conv8 = Weights([3, 3, 256, 512])
        b_conv8 = Biases([512])
        Conv8 = tf.nn.conv2d(pool3, W_conv8, strides=[1, 1, 1, 1], padding='SAME')
        actv8 = tf.nn.relu(Conv8 + b_conv8)

    print_activations(Conv8)
    print_activations(actv8)

    # Conv9  3x3x512
    with tf.name_scope('Conv9') as scope:
        W_conv9 = Weights([3, 3, 512, 512])
        b_conv9 = Biases([512])
        Conv9 = tf.nn.conv2d(actv8, W_conv9, strides=[1, 1, 1, 1], padding='SAME')
        actv9 = tf.nn.relu(Conv9 + b_conv9)

    print_activations(Conv9)
    print_activations(actv9)

    # Conv10  3x3x512
    with tf.name_scope('Conv10') as scope:
        W_conv10 = Weights([3, 3, 512, 512])
        b_conv10 = Biases([512])
        Conv10 = tf.nn.conv2d(actv9, W_conv10, strides=[1, 1, 1, 1], padding='SAME')
        actv10 = tf.nn.relu(Conv10 + b_conv10)

    print_activations(Conv10)
    print_activations(actv10)

    # pooling4  2x2
    with tf.name_scope('max_pooling4') as scope:
        pool4 = tf.nn.max_pool(actv10, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    print_activations(pool4)

    # Conv11  3x3x512
    with tf.name_scope('Conv11') as scope:
        W_conv11 = Weights([3, 3, 256, 512])
        b_conv11 = Biases([512])
        Conv11 = tf.nn.conv2d(pool4, W_conv11, strides=[1, 1, 1, 1], padding='SAME')
        actv11 = tf.nn.relu(Conv11 + b_conv11)

    print_activations(Conv11)
    print_activations(actv11)

    # Conv12  3x3x512
    with tf.name_scope('Conv12') as scope:
        W_conv12 = Weights([3, 3, 512, 512])
        b_conv12 = Biases([512])
        Conv12 = tf.nn.conv2d(actv11, W_conv12, strides=[1, 1, 1, 1], padding='SAME')
        actv12 = tf.nn.relu(Conv12 + b_conv12)

    print_activations(Conv12)
    print_activations(actv12)

    # Conv13  3x3x512
    with tf.name_scope('Conv13') as scope:
        W_conv13 = Weights([3, 3, 512, 512])
        b_conv13 = Biases([512])
        Conv13 = tf.nn.conv2d(actv9, W_conv13, strides=[1, 1, 1, 1], padding='SAME')
        actv13 = tf.nn.relu(Conv13 + b_conv13)

    print_activations(Conv13)
    print_activations(actv13)

    # pooling4  2x2
    with tf.name_scope('max_pooling5') as scope:
        pool5 = tf.nn.max_pool(actv13, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    print_activations(pool5)
    flatten = tf.reshape(pool5, [7*7*512])


    # Fully connnect 1
    with tf.name_scope('FCL1') as scope:
        W_FCL1 = Weights([4096, 4096])
        b_FCL1 = Biases([4096])
        FCL1 = tf.matmul(flatten, W_FCL1)+b_FCL1
        actv_FCL1 = tf.nn.relu(FCL1, name='actv')
    print_activations(actv_FCL1)


    # Fully connnect 2
    with tf.name_scope('FCL2') as scope:
        W_FCL2 = Weights([4096, 4096])
        b_FCL2 = Biases([4096])
        FCL2 = tf.matmul(actv_FCL1, W_FCL2)+b_FCL2
        actv_FCL2 = tf.nn.relu(FCL2, name='actv')
    print_activations(actv_FCL2)


    # Fully connnect 3
    with tf.name_scope('FCL3') as scope:
        W_FCL3 = Weights([4096, 1000])
        b_FCL3 = Biases([1000])
        FCL3 = tf.matmul(actv_FCL2, W_FCL3)+b_FCL3
        actv_FCL3 = tf.nn.relu(FCL3, name='actv')
    print_activations(actv_FCL3)

    prediction = tf.nn.sigmoid(actv_FCL3)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=prediction)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    prediction_2 = tf.nn.softmax(prediction)
    correct_predict = tf.equal(tf.argmax(prediction_2, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(40):
            for batch in range(batch_num):
                sess.run(train_step, feed_dict={X:train_x, Y:train_y, keep_drop:0.7})
            acc = sess.run(accuracy, feed_dict={X:test_x, Y:test_y, keep_drop:0.3})
        print(print("Iter: " + str(epoch) + ", acc: " + str(acc)))





