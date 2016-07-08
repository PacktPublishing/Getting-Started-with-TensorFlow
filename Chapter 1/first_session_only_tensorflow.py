#first_session_only_tensorflow.py

import tensorflow as tf

x = tf.constant(1, name='x')
y = tf.Variable(x+9,name='y')


model = tf.initialize_all_variables()

with tf.Session() as session:
    session.run(model)
    print(session.run(y))
