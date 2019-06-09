import tensorflow as tf

#fetch:一次运算多个op
input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)

add = tf.add(input2, input3)
mul = tf.multiply(input1, add)

with tf.Session() as sess:
    result = sess.run([mul, add])
    print(result)

print("Feed------------")
#feed:在运算op时传入值
#创建占位符
input4 = tf.placeholder(tf.float32)
input5 = tf.placeholder(tf.float32)
output = tf.multiply(input4, input5)

with tf.Session() as sess_feed:
    print(sess_feed.run(output, feed_dict={input4:[7.], input5:[2.]}))