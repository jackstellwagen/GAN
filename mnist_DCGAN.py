#import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import scipy.io
import scipy
import time
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)



batch_size = 35
num_steps = 800
vector_dim = 200
tf.reset_default_graph()

#print(data.shape)

#n_samp, n_input = data.shape
def noise_array():
    return np.random.normal(scale = 0.05, size =(500,27998))


perfect_data = np.indices((500,27998))[1]
sin = lambda t: np.sin(1/np.random.randint(600,1500) *t)
data = np.apply_along_axis(sin, 1, perfect_data) + noise_array()


noise = lambda t: np.random.normal(scale = 0.05) * t
vfunc = np.vectorize(noise)
#data = vfunc(data)


n_samp, n_input = data.shape
print(data.shape)



def generator(x, isTrain=True,reuse=False, batch_size=batch_size):
    with tf.variable_scope('Generator', reuse=reuse):
        dense1 = tf.layers.dense(x, units= 7*7 * 128,kernel_initializer =tf.contrib.layers.xavier_initializer())
        dense1 = tf.reshape(dense1, shape=[-1, 7, 7, 128])
        dense1 =tf.layers.batch_normalization(dense1,momentum=0.9, training =isTrain, epsilon=0.00001)
        dense1 = tf.nn.relu(dense1)

        conv1 = tf.layers.conv2d_transpose(dense1, 64, 5, strides=2, padding ="same")
        conv1 = tf.nn.relu(tf.layers.batch_normalization(conv1,momentum=0.9, training =isTrain, epsilon=0.00001))

        conv2 = tf.layers.conv2d_transpose(conv1, 1, 5, strides=2, padding ="same")
        conv2 = tf.nn.tanh(conv2)
        print(conv2.get_shape(), "conv2")
        return conv2

def discriminator(x,isTrain=True, reuse=False):
    with tf.variable_scope('Discriminator', reuse=reuse):


        conv1 = tf.layers.conv2d(x,128, 5,strides = 2, padding = "Same",kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d())
        conv1 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv1, momentum=0.9, training =isTrain, epsilon=0.00001))

        conv2 = tf.layers.conv2d(conv1, 256, 5,strides = 2,padding = "Same", kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d())
        conv2 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv2, momentum=0.9, training =isTrain, epsilon=0.00001))

        conv3 = tf.layers.conv2d(conv2, 1, 7,strides = 1, padding = "Valid",kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d())
        out = tf.nn.sigmoid(conv3)
        print(conv3.get_shape(), "conv3")
        """
        conv3 = tf.layers.conv2d(conv2, 512, 5,stride = 2, padding = "Same",kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d())
        conv3 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv3, momentum=0.9, training =isTrain, epsilon=0.00001))

        conv4 = tf.layers.conv2d(conv3, 1024, 4, stride = 2,padding = "Same",kernel_initializer =tf.contrib.layers.xavier_initializer_conv2d())
        conv4 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv4, momentum=0.9, training =isTrain, epsilon=0.00001))

        conv5 = tf.layers.conv2d(conv4, 1, 4,stride = 1, padding = "Same", kernel_initializer =tf.contrib.layers.xavier_initializer_conv2d())
        out = tf.nn.sigmoid(conv5)
        """

        return out,conv3




random_vector = tf.placeholder(tf.float32,shape=[None,vector_dim])
real_image_input = tf.placeholder(tf.float32, shape=[None,28,28, 1])


isTrain = tf.placeholder(dtype=tf.bool)

gen_sample = generator(random_vector, isTrain, batch_size=batch_size)

disc_real,disc_real_logits = discriminator(real_image_input,isTrain)
disc_fake,disc_fake_logits = discriminator(gen_sample,isTrain, reuse=True)


gan_model = discriminator(gen_sample,reuse=True)


gen_target = tf.placeholder(tf.int32, shape=[None])
disc_target = tf.placeholder(tf.int32, shape=[None])

disc_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=disc_real_logits, labels=tf.ones_like(disc_real_logits)))
disc_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=disc_fake_logits, labels=tf.zeros_like(disc_fake_logits)))
disc_loss = disc_loss_real + disc_loss_fake
gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=disc_fake_logits, labels=tf.ones_like(disc_fake_logits)))

optimizer_gen = tf.train.AdamOptimizer(learning_rate=0.002, beta1=0.5)
optimizer_disc = tf.train.AdamOptimizer(learning_rate=0.002, beta1= 0.5)


gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')


update_ops_gen = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope = "Generator")
update_ops_disc = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="Discriminator")

with tf.control_dependencies(update_ops_gen):
     train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)

with tf.control_dependencies(update_ops_disc):
    train_disc = optimizer_gen.minimize(disc_loss, var_list=disc_vars)

init = tf.global_variables_initializer()


saver = tf.train.Saver()
s = time.time()
with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess,"/home/jack/caltech_research/mnist_DCGAN/mnist_DCGAN.ckpt")
    for i in range(1, num_steps+1):
        epoch_x, _ = mnist.train.next_batch(batch_size)
        epoch_x = np.reshape(epoch_x, newshape=[-1, 28, 28, 1])
        z = np.random.uniform(-1.0, 1.0, size=[batch_size, vector_dim])

       	dl,_,dlr,dlf = sess.run([disc_loss,train_disc,disc_loss_real, disc_loss_fake], feed_dict = {real_image_input:epoch_x, random_vector:z, isTrain:True})
        gl, _ = sess.run([gen_loss,train_gen], feed_dict = {random_vector:z,isTrain:True})

        if i % 100 == 0 or i == 1:
            print('Step %i: Generator Loss: %f, Discriminator Loss: %f' % (i, gl, dl))
            print("DLR:",dlr,",", "DLF:", dlf)
    time_taken = time.time()-s
    print(time_taken, "time taken")
    np.save("time.npy", np.zeros(shape=[1])+time_taken)
    save_path = saver.save(sess, "/home/jack/caltech_research/mnist_DCGAN/mnist_DCGAN.ckpt")


    epoch_x, _ = mnist.train.next_batch(batch_size)
    epoch_x = np.reshape(epoch_x, newshape=[-1, 28, 28, 1])
    z = np.random.uniform(-1., 1., size=[batch_size, vector_dim])
    samp  = sess.run(gen_sample, feed_dict={random_vector: z, isTrain:False})

    g = sess.run(disc_real, feed_dict={real_image_input: epoch_x, isTrain:True})
    o = sess.run(disc_real, feed_dict={real_image_input: samp, isTrain:False})

    #np.save("disc_output_fake.npy",o)
    #np.save("disc_output.npy", g)
    np.save("mnist_output.npy", samp)
    np.save("epoch_x.npy", epoch_x)
