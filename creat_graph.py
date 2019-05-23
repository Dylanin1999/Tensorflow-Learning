import tensorflow as tf

#创建一个常量node
m1 = tf.constant([[3, 3]])

#创建一个常量node
m2 = tf.constant([[2], [3]])

#创建一个矩阵乘法operation，把m1和m2传入
product = tf.matmul(m1, m2)

#定义一个会话，启动默认图
sess = tf.Session()
#调用sess的run方法来执行operation
result = sess.run(product)

print(result)
sess.close()

"""
#不用自己手动创建会话，程序会自动创建和销毁
with tf.Session() as sess:
    result = sess.run(product)
    print(result)
"""