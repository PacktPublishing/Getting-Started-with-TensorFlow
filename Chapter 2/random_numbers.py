import tensorflow as tf
import matplotlib.pyplot as plt

# Create a tensor of shape [100] consisting of random normal values, with mean
# 0 and standard deviation 2.
norm = tf.random_normal([100], mean=0, stddev=2)
with tf.Session() as session:
    plt.hist(norm.eval(),normed=True)
    plt.show()  


uniform = tf.random_uniform([100],minval=0,maxval=1,dtype=tf.float32)
with tf.Session() as session:
    print uniform.eval()
    plt.hist(uniform.eval(),normed=True)
    plt.show() 


uniform_with_seed = tf.random_uniform([1], seed=1)
uniform_without_seed = tf.random_uniform([1])

# Repeatedly running this block with the same graph will generate the same
# sequence of values for 'a', but different sequences of values for 'b'.
print("First Run")
with tf.Session() as first_session:
  print("uniform with (seed = 1) = {}"\
        .format(first_session.run(uniform_with_seed)))  
  print("uniform with (seed = 1) = {}"\
        .format(first_session.run(uniform_with_seed)))
  print("uniform without seed = {}"\
        .format(first_session.run(uniform_without_seed)))  
  print("uniform without seed = {}"\
        .format(first_session.run(uniform_without_seed)))  

print("Second Run")
with tf.Session() as second_session:
  print("uniform with (seed = 1) = {}"\
        .format(second_session.run(uniform_with_seed)))  
  print("uniform with (seed = 1) = {}"\
        .format(second_session.run(uniform_with_seed)))  
  print("uniform without seed = {}"\
        .format(second_session.run(uniform_without_seed)))  
  print("uniform without seed = {}"\
        .format(second_session.run(uniform_without_seed)))



import tensorflow as tf

trials = 100
hits = 0
x = tf.random_uniform([1],minval=-1,maxval=1,dtype=tf.float32)
y = tf.random_uniform([1],minval=-1,maxval=1,dtype=tf.float32)
pi = []
sess = tf.Session()
with sess.as_default():
    for i in range(1,trials):
        for j in range(1,trials):
            if x.eval()**2 + y.eval()**2 < 1 :
                hits = hits + 1
                pi.append((4 * float(hits) / i)/trials)  

plt.plot(pi)
plt.show()
