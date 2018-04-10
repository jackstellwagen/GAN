
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import scipy.io
import scipy
import time


#data = np.load("/Users/Jack/Desktop/neuraldev/AdCo/matrix_GAN_2D.npy")
data = np.load("/home/jack/caltech_research/neuraldev/GAN_data.npy")
#data = data.tolist().toarray()
data = data.transpose()

batch_size = 26
num_steps = 600
vector_dim = 200
tf.reset_default_graph()

n_samp, n_input = data.shape

def generator(x, reuse=False):
    with tf.variable_scope('Generator', reuse=reuse):
        x = tf.layers.dense(x, units= 332 * 1 * 64)
        x = tf.nn.tanh(x)
        x = tf.reshape(x, shape=[-1, 332, 1, 64])


        conv1 = tf.layers.conv2d_transpose(x, 64, [30,1], strides=[3,1])

        conv2 = tf.layers.conv2d_transpose(conv1, 32, [30,1], strides=[3,1])#, padding="same")

        conv3 = tf.layers.conv2d_transpose(conv2, 8, [30,1], strides=[3,1])#, padding="same")

        conv4 = tf.layers.conv2d_transpose(conv3, 1, [30,1], strides=[3,1])# padding="same")

        conv5 = tf.layers.conv2d_transpose(conv4, 1, [27,1], strides=[1,1])# padding="same")
        conv5 = tf.nn.relu(conv5)
        #conv5 = tf.nn.sigmoid(conv5)
        conv5 = tf.squeeze(conv5, axis = 2)
        print(conv5.get_shape())
        #print(conv1.get_shape())
        """
        print(conv1.get_shape(), 'o1')
        print(conv2.get_shape(), 'o2')
        print(conv3.get_shape(), 'o3')
        print(conv4.get_shape(), 'o4')
        print(conv5.get_shape(), 'o5')

        """
        return conv5

        """
        x = tf.layers.dense(x, units=6 * 6 * 128)
        #print(x.get_shape(), 'o')
        x = tf.nn.tanh(x)
        #print(x.get_shape())
        x = tf.layers.dense(x, units=6 * 19 * 128)
        #print(x.get_shape())
        x = tf.nn.tanh(x)
        x = tf.reshape(x, shape=[-1, 6, 19, 128])
        #conv1 = tf.layers.conv2d_transpose(x, 64, 2, strides=2)
        #print(conv1.get_shape())
        conv1 = tf.layers.conv2d_transpose(x, 64, 4, strides=2) #(6,19)-->(14,40)
        #print(conv1.get_shape())
        conv2 = tf.layers.conv2d_transpose(conv1, 1, [6,5], strides=2) #(14,40)-->(32*83)
        #print(conv2.get_shape())
        #conv2 = tf.nn.sigmoid(conv2)
        conv2 = tf.nn.relu(conv2)
        """

def discriminator(x, reuse=False):
    with tf.variable_scope('Discriminator', reuse=reuse):
        # Typical convolutional neural network to classify images.
        #print(x.get_shape(),'i')
        #conv1 = conv1d()
        conv1 = tf.layers.conv1d(x,16, 30, padding = "Same")
        conv1 = tf.nn.tanh(conv1)
        conv1_pool = tf.layers.average_pooling1d(conv1, 3, 3,  padding = "Same")
        conv2 = tf.layers.conv1d(conv1_pool, 32, 30,  padding = "Same")
        conv2 = tf.nn.tanh(conv2)
        conv2_pool = tf.layers.average_pooling1d(conv2, 3, 3,  padding = "Same")
        conv3 = tf.layers.conv1d(conv2_pool, 64, 30, padding = "Same")
        conv3 = tf.nn.tanh(conv3)
        conv3_pool = tf.layers.average_pooling1d(conv3, 3, 3,  padding = "Same")
        conv4 = tf.layers.conv1d(conv3_pool, 64, 30, padding = "Same")
        conv4 = tf.nn.tanh(conv4)
        conv4_pool = tf.layers.average_pooling1d(conv4, 3, 3,  padding = "Same")
        flatten = tf.contrib.layers.flatten(conv4_pool)
        dense1 = tf.layers.dense(flatten, 2048)
        dense1 = tf.nn.tanh(dense1)
        dense2 = tf.layers.dense(dense1, 512)
        dense2 = tf.nn.tanh(dense2)
        dense3 = tf.layers.dense(dense2, 2)
        dense3 = tf.nn.sigmoid(dense3)
        #dense3 = tf.nn.softmax(dense3)
        """
        print(x.get_shape(),"xshape")
        print(conv1.get_shape(),'conv1')
        print(conv1_pool.get_shape(), "pool1")
        print(conv2.get_shape(),'conv2')
        print(conv2_pool.get_shape(), "pool2")
        print(conv3.get_shape(),'conv3')
        print(conv3_pool.get_shape(), "pool3")
        print(conv4.get_shape(),'conv4')
        print(conv4_pool.get_shape(), "pool4")
        print(flatten.get_shape(), "flatten")
        print(dense1.get_shape(), 'dense1')
        print(dense2.get_shape(), 'dense2')
        print(dense3.get_shape(), 'dense3')
        """



        #print(conv2.get_shape(), "conv2")

        """
        print(x.get_shape(),"xshape")
        conv1 = tf.layers.conv2d(x, 64, [5,4])
        print(conv1.get_shape(),'o')
        conv1 = tf.nn.tanh(conv1)
        conv1_pool = tf.layers.average_pooling2d(conv1, 2, 2)
        #print(conv1_pool.get_shape(),'pool')
        conv2 = tf.layers.conv2d(conv1_pool, 128, 5)
        #print(conv2.get_shape(),'o2')
        conv2 = tf.nn.tanh(conv2)
        conv2_pool= tf.layers.average_pooling2d(conv2, 2, 2)
        #print(conv2_pool.get_shape(),'pool2')
        conv2 = tf.contrib.layers.flatten(conv2_pool)
        #print(conv2.get_shape(),'flatten')
        conv2 = tf.layers.dense(conv2, 1024)
        #print(conv2_pool.get_shape(),'dense')
        conv2 = tf.nn.tanh(conv2)
        #print(conv2.get_shape())
        # Output 2 classes: Real and Fake images
        x = tf.layers.dense(conv2, 2)
        #print(x.get_shape(),'final out')
        """


    return dense3

random_vector = tf.placeholder(tf.float32,shape=[None,vector_dim])
real_image_input = tf.placeholder(tf.float32, shape=[None,27998, 1])

gen_sample = generator(random_vector)

disc_real = discriminator(real_image_input)
disc_fake = discriminator(gen_sample, reuse=True)
disc_concat = tf.concat([disc_real,disc_fake],axis=0)
print(disc_concat.get_shape(),'concat')

gan_model = discriminator(gen_sample,reuse=True)

"""
disc_target = tf.placeholder(tf.int32, shape=[None])
gen_target = tf.placeholder(tf.int32, shape=[None])
"""
#defining the target values for discriminator output
disc_target = np.concatenate(
            [np.ones([batch_size]), np.zeros([batch_size])], axis=0)
reverse =np.flip(disc_target,axis =0)
disc_target = np.stack((disc_target,reverse), axis =0).transpose()
disc_target = tf.convert_to_tensor(disc_target, dtype = tf.float32)

gen_out = np.zeros([batch_size])
disc_out = np.ones([batch_size])
gen_target = np.stack((disc_out,gen_out), axis =0).transpose()
gen_target = tf.convert_to_tensor(gen_target, dtype = tf.float32)

"""
disc_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=disc_concat, labels=disc_target))#[:,0]
#print(tf.nn.sparse_softmax_cross_entropy_with_logits(
 #   logits=disc_concat, labels=disc_target))
gen_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=gan_model, labels=gen_target))#[:,1]
"""
disc_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=disc_concat, labels=disc_target))#[:,0]
#print(tf.nn.sparse_softmax_cross_entropy_with_logits(
 #   logits=disc_concat, labels=disc_target))
gen_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=gan_model, labels=gen_target))#[:,1]


optimizer_gen = tf.train.AdamOptimizer(learning_rate=0.0001)
optimizer_disc = tf.train.AdamOptimizer(learning_rate=0.0001)


gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')


train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars)



init = tf.global_variables_initializer()



saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, "/home/jack/caltech_research/cell_data_GAN_network/cell_data_GAN_trained.ckpt")
    z = np.random.uniform(-1., 1., size=[200, vector_dim])
    g = sess.run(gen_sample, feed_dict={random_vector: z})
    #g = np.load("/Users/Jack/caltech_research/network_output/linear_cell_GAN.n$
    #g = np.reshape(data, newshape=[-1, 27998, 1])
    #g = g[:250]
    x = sess.run(disc_real, feed_dict={real_image_input: g})

    np.save("/home/jack/caltech_research/network_output/linear_cell_GAN.npy", g)
    np.save("/home/jack/caltech_research/network_output/linear_cell_GAN_disc.npy",x)


