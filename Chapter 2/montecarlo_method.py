import tensorflow as tf

trials = 100
hits = 0
x = tf.random_uniform([1],minval=0,maxval=1,dtype=tf.float32)
y = tf.random_uniform([1],minval=0,maxval=1,dtype=tf.float32)

sess = tf.Session()
with sess.as_default():
    for i in range(1,trials):
        for j in range(1,trials):
            if x.eval()**2 + y.eval()**2 < 1 :
                hits = hits + 1
        print (4 * float(hits) / i)/trials  





