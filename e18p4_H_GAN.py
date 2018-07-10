
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import scipy.io
import scipy
import time


batch_size = 35
num_steps = 2000
vector_dim = 8
tf.reset_default_graph()

#print(data.shape)

#n_samp, n_input = data.shape

def noise_array():
    return np.random.normal(scale = 0.05, size =(500,1024))



from numpy import genfromtxt
data = genfromtxt('e18p4_H.csv', delimiter=',')

data = ((data/(data.max())) *2)-1
data = data.transpose()

print(data.max(), data.min(), "data max, min")
n_samp, n_input = data.shape
print(data.shape)
#n_samp, n_input = data.shape


def generator(x, isTrain=True,reuse=False, batch_size=batch_size):
    with tf.variable_scope('Generator', reuse=reuse):
        
        x = tf.layers.dense(x, units= 6 * 128 ,kernel_initializer =tf.contrib.layers.xavier_initializer())#332
        x = tf.reshape(x, shape=[-1,6, 1, 128])
        x = tf.layers.batch_normalization(x,momentum=0.9, training =isTrain, epsilon=0.00001)
        x = tf.nn.relu(x)
        print(x.get_shape(), "x")
        
        conv1 = tf.layers.conv2d_transpose(x, 64, [5,1], strides=[2,1],kernel_initializer = tf.contrib.layers.xavier_initializer(), padding= "same")
        conv1 = tf.nn.relu(tf.layers.batch_normalization(conv1, training =isTrain,momentum=0.9,epsilon=0.00001))
        print(conv1.get_shape(), "conv1")

        conv2 = tf.layers.conv2d_transpose(conv1, 1, [5,1], strides=[2,1],kernel_initializer = tf.contrib.layers.xavier_initializer(), padding = "same")
        #conv2 = tf.nn.relu(tf.layers.batch_normalization(conv2, training =isTrain,momentum=0.9,epsilon=0.00001))
        conv2 = tf.squeeze(conv2, axis = 2)
        conv2 = tf.nn.tanh(conv2)
        return conv2


        #conv3 = tf.layers.conv2d_transpose(conv2, 1, [6,1], strides=[2,1],kernel_initializer = tf.contrib.layers.xavier_initializer(), padding = "same")
        #conv3 = tf.nn.relu(tf.layers.batch_normalization(conv3, training =isTrain,momentum=0.9,epsilon=0.00001))
        """
        x = tf.layers.dense(x, units= 3 * 128 ,kernel_initializer =tf.contrib.layers.xavier_initializer())#332
        x = tf.reshape(x, shape=[-1,3, 1, 128])
        x = tf.layers.batch_normalization(x,momentum=0.9, training =isTrain, epsilon=0.00001)
        x = tf.nn.relu(x)
        print(x.get_shape(), "x")

        conv1 = tf.layers.conv2d_transpose(x, 64, [5,1], strides=[2,1],kernel_initializer = tf.contrib.layers.xavier_initializer(), padding= "same")
        conv1 = tf.nn.relu(tf.layers.batch_normalization(conv1, training =isTrain,momentum=0.9,epsilon=0.00001))
        print(conv1.get_shape(), "conv1")

        conv2 = tf.layers.conv2d_transpose(conv1, 32, [5,1], strides=[2,1],kernel_initializer = tf.contrib.layers.xavier_initializer(), padding = "same")
        conv2 = tf.nn.relu(tf.layers.batch_normalization(conv2, training =isTrain,momentum=0.9,epsilon=0.00001))
        #conv2 = tf.squeeze(conv2, axis = 2)
        #conv2 = tf.nn.tanh(conv2)
        


        conv3 = tf.layers.conv2d_transpose(conv2, 1, [5,1], strides=[2,1],kernel_initializer = tf.contrib.layers.xavier_initializer(), padding = "same")
        #conv3 = tf.nn.relu(tf.layers.batch_normalization(conv3, training =isTrain,momentum=0.9,epsilon=0.00001))
        conv3 = tf.squeeze(conv3, axis =2)
        conv3 = tf.nn.tanh(conv3)
        return conv3
        """
        """

        conv4 = tf.layers.conv2d_transpose(conv3, 4, [10,1], strides=[2,1],kernel_initializer = tf.contrib.layers.xavier_initializer(), padding ="same")
        conv4 = tf.nn.relu(tf.layers.batch_normalization(conv4, training =isTrain,momentum=0.9,epsilon=0.00001))
        print(conv4.get_shape(), "conv4")
        
        conv5 = tf.layers.conv2d_transpose(conv4, 1, [10,1], strides=[2,1],kernel_initializer = tf.contrib.layers.xavier_initializer(), padding ="same")
        #conv5 = tf.nn.relu(tf.layers.batch_normalization(conv5, training =isTrain,momentum=0.9,epsilon=0.0001))

        #conv5 = tf.layers.conv2d_transpose(conv4, 1, [27,1], strides=[1,1],kernel_initializer = tf.contrib.layers.xavier_initializer())# padding="same")
        """
        conv3 = tf.squeeze(conv3, axis = 2)
        conv3 = tf.nn.tanh(conv3)
        print(conv3.get_shape(), "conv3 shape")
        return conv3

def discriminator(x,isTrain=True, reuse=False):
    with tf.variable_scope('Discriminator', reuse=reuse):
        conv1 = tf.layers.conv1d(x,128, 5,strides=2, padding = "Same",kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d())
        conv1 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv1, momentum=0.9, training =isTrain, epsilon=0.00001))

        conv2 = tf.layers.conv1d(conv1, 256, 5,strides =2,padding = "Same", kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d())
        conv2 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv2, momentum=0.9, training =isTrain, epsilon=0.00001))
 
        conv3 = tf.layers.conv1d(conv2, 1, 6,strides =1, padding = "VALID",kernel_initializer =tf.contrib.layers.xavier_initializer_conv2d())
        """
        conv3 = tf.layers.conv1d(conv1, 512, 5,strides =2,padding = "Same", kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d())
        conv3 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv3, momentum=0.9, training =isTrain, epsilon=0.00001))

        conv4 = tf.layers.conv1d(conv3, 1, 3,strides =1, padding = "VALID",kernel_initializer =tf.contrib.layers.xavier_initializer_conv2d())
        out = tf.nn.sigmoid(conv4)
        return out, conv4
        """
               
        out = tf.nn.sigmoid(conv3)
        return out, conv3
        
        
        conv3 = tf.layers.conv1d(conv2, 64, 6,strides =2, padding = "Same",kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d())
        conv3 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv3, momentum=0.9, training =isTrain, epsilon=0.00001))
       
        print(conv3.get_shape(), "disc conv3")
        conv4 = tf.layers.conv1d(conv3, 1, 3,strides =1, padding = "VALID",kernel_initializer =tf.contrib.layers.xavier_initializer_conv2d())
        out = tf.nn.sigmoid(conv4)
        return out, conv4
        
        """ 
        conv4 = tf.layers.conv1d(conv3, 32, 10,strides =2, padding = "Same",kernel_initializer =tf.contrib.layers.xavier_initializer_conv2d())
        conv4 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv4, momentum=0.9, training =isTrain, epsilon=0.00001))

       
        conv5 = tf.layers.conv1d(conv4, 64, 10,strides =2, padding = "Same",kernel_initializer =tf.contrib.layers.xavier_initializer_conv2d())
        conv5 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv5, momentum=0.9, training =isTrain, epsilon=0.00001))

        conv6 = tf.layers.conv1d(conv5, 1, 32,strides =1, padding = "VALID",kernel_initializer =tf.contrib.layers.xavier_initializer_conv2d())
        out = tf.nn.sigmoid(conv6)
        print(out.get_shape(), "out shape")
        

        return out, conv6
        """

#return dense3

random_vector = tf.placeholder(tf.float32,shape=[None,vector_dim])
real_image_input = tf.placeholder(tf.float32, shape=[None,24, 1])


isTrain = tf.placeholder(dtype=tf.bool)

gen_sample = generator(random_vector, isTrain, batch_size=batch_size)


disc_real,disc_real_logits = discriminator(real_image_input,isTrain)
disc_fake,disc_fake_logits = discriminator(gen_sample,isTrain, reuse=True)
"""
disc_real,disc_real_logits = discriminator(real_image_input,isTrain=False)
disc_fake,disc_fake_logits = discriminator(gen_sample,isTrain=False, reuse=True)
"""
#disc_fake2,disc_fake_logits2 = discriminator(gen_sample,isTrain=False, reuse=True)


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

#extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#with tf.control_dependencies(extra_update_ops):

optimizer_gen = tf.train.AdamOptimizer(learning_rate=0.002, beta1=0.5)
optimizer_disc = tf.train.AdamOptimizer(learning_rate=0.002, beta1= 0.5)


gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')

#gen  = generator(random_vector,batch_size=100,reuse=True)

#print(gen_vars)
#extra_update_ops_disc = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope ="Discriminator")
#extra_update_ops_gen = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope ="Generator")

#train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
#train_disc = optimizer_gen.minimize(disc_loss, var_list=disc_vars)
update_ops_gen = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope = "Generator")
update_ops_disc = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="Discriminator")

with tf.control_dependencies(update_ops_gen):
     train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)

with tf.control_dependencies(update_ops_disc):
    train_disc = optimizer_gen.minimize(disc_loss, var_list=disc_vars)

"""
def train(lossg, lossd):

    optimizer_gen = tf.train.AdamOptimizer(learning_rate=0.002, beta1=0.5)
    optimizer_disc = tf.train.AdamOptimizer(learning_rate=0.002, beta1= 0.5)

    train_g = optimizer_gen.minimize(lossg, var_list=gen_vars)
    train_d = optimizer_disc.minimize(lossd, var_list=disc_vars)
    with tf.control_dependencies([train_g, train_d]):
        return tf.no_op(name="train")
train_op = train(gen_loss,disc_loss)

def traind(dloss):

    #optimizer_gen = tf.train.AdamOptimizer(learning_rate=0.002, beta1=0.5)
    optimizer_disc = tf.train.AdamOptimizer(learning_rate=0.002, beta1= 0.5)

    #train_g = optimizer_gen.minimize(loss, var_list=gen_vars)
    train_d = optimizer_disc.minimize(dloss, var_list=disc_vars)
    with tf.control_dependencies( train_d):
        return tf.no_op(name="traind")

def traing(gloss):

    optimizer_gen = tf.train.AdamOptimizer(learning_rate=0.002, beta1=0.5)
    #optimizer_disc = tf.train.AdamOptimizer(learning_rate=0.002, beta1= 0.5)

    train_g = optimizer_gen.minimize(gloss, var_list=gen_vars)
    #train_d = optimizer_disc.minimize(loss, var_list=disc_vars)
    with tf.control_dependencies(train_g):
        return tf.no_op(name="traing")
#train_gen = train(gen_loss)
#train_disc = train(disc_loss)

with tf.control_dependencies(extra_update_ops_gen):
     train_gen = optimizer_gen.minimize(gen_loss)#, var_list=gen_vars)

#train_gen = optimizer_gen.maximize(gen_loss, var_list=gen_vars)

with tf.control_dependencies(extra_update_ops_disc):
     train_disc = optimizer_disc.minimize(disc_loss)#, var_list=disc_vars)
"""

init = tf.global_variables_initializer()

						
saver = tf.train.Saver()
s = time.time()
with tf.Session() as sess:
    #start = time.time()
    sess.run(init)
    gloss = np.zeros(shape=[1])
    dl = 0
    dlr = 0
    dlf = 0
    saver.restore(sess, "/home/jack/caltech_research/e18p4_H_GAN_network2/e18p4_H_GAN_trained.ckpt")
    for i in range(1, num_steps+1):
        """
        if i% 50 == 0:# and i!=500:
             start = time.time()
             #data = np.apply_along_axis(sin, 1, perfect_data) + noise_array()
             data = gen_data()
             np.save("data.npy", data)
             print(time.time()-start, "time")
        """
        sample = np.random.randint(n_samp, size=batch_size)
        epoch_x = data[sample,:]
        epoch_x = np.reshape(epoch_x, newshape=[-1, 24, 1])
        z = np.random.normal(0, 0.5, size=[batch_size, vector_dim])
       	dl,_,dlr,dlf = sess.run([disc_loss,train_disc,disc_loss_real, disc_loss_fake], feed_dict = {real_image_input:epoch_x, random_vector:z, isTrain:True})
        gl, _ = sess.run([gen_loss,train_gen], feed_dict = {random_vector:z,isTrain:True})
        
        if i == 1:
           z = np.random.normal(0, 0.5, size=[40000, vector_dim])
           samp  = sess.run(gen_sample, feed_dict={random_vector: z, isTrain:False})
           np.save("e18p4_H_incremental_training/e18p4_H_1epoch.npy",samp)
        if i == 10:
           z = np.random.normal(0, 0.5, size=[40000, vector_dim])
           samp  = sess.run(gen_sample, feed_dict={random_vector: z, isTrain:False})
           np.save("e18p4_H_incremental_training/e18p4_H_10epoch.npy",samp)
        if i == 100:
           z = np.random.normal(0, 0.5, size=[40000, vector_dim])
           samp  = sess.run(gen_sample, feed_dict={random_vector: z, isTrain:False})
           np.save("e18p4_H_incremental_training/e18p4_H_100epoch.npy", samp)
        if i == 1000:
           z = np.random.normal(0, 0.5, size=[40000, vector_dim])
           samp  = sess.run(gen_sample, feed_dict={random_vector: z, isTrain:False})
           np.save("e18p4_H_incremental_training/e18p4_H_1000epoch.npy",samp)
        
        if i % 100 == 0 or i == 1:
            print('Step %i: Generator Loss: %f, Discriminator Loss: %f' % (i, gl, dl))
            print("DLR:",dlr,",", "DLF:", dlf)
    save_path = saver.save(sess, "/home/jack/caltech_research/e18p4_H_GAN_network2/e18p4_H_GAN_trained.ckpt")
    #np.save("gloss.npy", gloss)
    time_taken = time.time()-s
    print(time_taken, "time taken")
    np.save("time.npy", np.zeros(shape=[1])+time_taken)
    sample = np.random.randint(n_samp, size=batch_size)
    epoch_x = data[sample,:]
    epoch_x = np.reshape(epoch_x, newshape=[-1, 24,1])
    #print(epoch_x)
    z = np.random.normal(0, 0.5, size=[40000, vector_dim])
    z2 = np.random.uniform(-1., 1., size=[batch_size, vector_dim])
    #z[34] = 0
    #z[33] = 0.2
    #z[32] = -0.2
    #gen = generator(random_vector,batch_size=100,reuse=True)
    samp  = sess.run(gen_sample, feed_dict={random_vector: z, isTrain:False})
    samp2  = sess.run(gen_sample, feed_dict={random_vector: z2, isTrain:False})    

    g = sess.run(disc_real, feed_dict={real_image_input: epoch_x, isTrain:False})
    o = sess.run(disc_real, feed_dict={real_image_input: samp, isTrain:False})
    #o = epoch_x
    np.save("disc_output_fake.npy",o)    
    np.save("disc_output.npy", g)
    np.save("e18p4_H_incremental_training/generator_output.npy", samp)
    np.save("generator_output2.npy", samp2)
    #np.save("generator_output2.npy", samp2)

def unnormalize(data, orig):
   data += 1
   data = data/2
   data = data*orig.max()
