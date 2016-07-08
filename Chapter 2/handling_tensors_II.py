#handling tensors second example


#STEP 1 --- PREPARE THE DATA
import matplotlib.image as mp_image
filename = "packt.jpeg"
input_image = mp_image.imread(filename)

#dimension
print 'input dim = {}'.format(input_image.ndim)
#shape
print 'input shape = {}'.format(input_image.shape)

height,width,depth= input_image.shape

import matplotlib.pyplot as plt
plt.imshow(input_image)
plt.show()

import tensorflow as tf

x = tf.Variable(input_image,name='x')
model = tf.initialize_all_variables()

with tf.Session() as session:
    x = tf.transpose(x, perm=[1,0,2])
    session.run(model)
    result=session.run(x)

plt.imshow(result)
plt.show()

