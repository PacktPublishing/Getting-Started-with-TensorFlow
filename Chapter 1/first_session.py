#first_session.py

#a simple Python code
x = 1
y = x + 9
print(y)

#....and the tensorflow translation of the previous code
import tensorflow as tf

x = tf.constant(1, name='x')
y = tf.Variable(x+9,name='y')
print(y)
