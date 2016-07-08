import numpy as np

tensor_1d = np.array([1.3,1,4.0,23.99])

print tensor_1d

print tensor_1d[0]

print tensor_1d[2]

import tensorflow as tf

tf_tensor = tf.convert_to_tensor(tensor_1d, dtype=tf.float64)

with tf.Session() as sess:
    print sess.run(tf_tensor)
    print sess.run(tf_tensor[0])
    print sess.run(tf_tensor[2])


tensor_2d=np.array([(1,2,3,4),(4,5,6,7),(8,9,10,11),(12,13,14,15)])

print tensor_2d
print tensor_2d[3][3]
print tensor_2d[0:2,0:2]

tf_tensor=tf.placeholder("float64",tensor_2d,name='x')
with tf.Session() as sess:
    print sess.run(x)
