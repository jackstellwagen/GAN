import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import scipy.io
import scipy
import time
#from fake_data_func import gen_fake_data

#data = np.load("/Users/Jack/Desktop/neuraldev/AdCo/matrix_GAN_2D.npy")
#data = np.load("/home/jack/caltech_research/neuraldev/GAN_data.npy")
#data = np.load("/home/jack/caltech_research/neuraldev/GAN_data_normalized_-1to1.npy")
#data = np.load("/home/jack/caltech_research/neuraldev/fake_GAN_data.npy")
#data = gen_fake_data()
#data = data.tolist().toarray()
#data = data.transpose()

#test_data = np.random.normal(loc=-0.1,scale=0.4,size=(100,27998,1))


batch_size = 35
num_steps = 200
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




#vfunc = np.vectorize(sin)

#perfect_data = np.zeros(shape=(500,27998))

#perfect_data[:,8000:16000] = 0.7
#perfect_data[:,16000:] = -0.7
"""
perfect_data[0:250,:12500] = 0.7
perfect_data[250:, 12500:] = -0.2 

sin_wave = np.zeros(shape=(27998))
for i in range(len(sin_wave)):
    sin_wave[i] = np.sin(i/1000)
perfect_data[:,:] = sin_wave
print(perfect_data.shape, "sin_wave")
np.save("perfect_data.npy", perfect_data)    
"""
#data = perfect_data + noise_array()

n_samp, n_input = data.shape
print(data.shape)


with tf.variable_scope('Generator', reuse=True):
     bconv1 = tf.Variable(tf.zeros([64]), name ="bconv1")
     bconv2 = tf.Variable(tf.zeros([32]), name ="bconv2")
     bconv3 = tf.Variable(tf.zeros([16]), name ="bconv3")
     bconv4 = tf.Variable(tf.zeros([8]), name ="bconv4")
     bconv5 = tf.Variable(tf.zeros([1]), name ="bconv5")
     wconv1 = tf.Variable(tf.truncated_normal([30,64,64]), name ="wconv1")
     wconv2 = tf.Variable(tf.truncated_normal([30,32,64]), name ="wconv2")
     wconv3 = tf.Variable(tf.truncated_normal([30,16,32]), name ="wconv3")
     wconv4 = tf.Variable(tf.truncated_normal([30,8,16]), name ="wconv4")
     wconv5 = tf.Variable(tf.truncated_normal([54,1,8]), name ="wconv5")

def generator(x, isTrain=True,reuse=False, batch_size=batch_size):
    with tf.variable_scope('Generator', reuse=reuse):
        x = tf.layers.dense(x, units= 345 * 64,activation = tf.identity,kernel_initializer =tf.contrib.layers.xavier_initializer())
        #x = tf.identity(x)
        x = tf.reshape(x, shape=[-1, 345, 64])
        x =tf.layers.batch_normalization(x,momentum=0.9, training =isTrain, epsilon=0.0001)#,name = "g_bn_d1")
        x = tf.nn.relu(x)
        
        #wconv1 = tf.Variable(tf.truncated_normal([30,64,64]), name ="wconv1")
        conv1 = tf.contrib.nn.conv1d_transpose(x,wconv1,[batch_size,1035,64], stride=3,padding ="SAME")
        #bconv1 = tf.Variable(tf.zeros([64]), name ="bconv1")
        conv1 = tf.nn.bias_add(conv1, bconv1)
        conv1 = tf.nn.relu(tf.layers.batch_normalization(conv1,momentum=0.9, training =isTrain,epsilon=0.0001))#,name="g_bn_conv1"))
        print(conv1.get_shape(),"conv1")     

        #wconv2 = tf.Variable(tf.truncated_normal([30,32,64]), name ="wconv2")
        conv2 = tf.contrib.nn.conv1d_transpose(conv1, wconv2,[batch_size, 3105,32], stride=3, padding="SAME")
        #bconv2 = tf.Variable(tf.zeros([32]), name ="bconv2")
        conv2 = tf.nn.bias_add(conv2, bconv2)
        conv2 = tf.nn.relu(tf.layers.batch_normalization(conv2,momentum=0.9,training =isTrain,epsilon=0.0001))#,name="g_bn_conv2"))
        print(conv2.get_shape(),"conv2")

        #wconv3 = tf.Variable(tf.truncated_normal([30,16,32]), name ="wconv3")
        conv3 = tf.contrib.nn.conv1d_transpose(conv2, wconv3,[batch_size, 9315,16], stride=3, padding="SAME")
        #bconv3 = tf.Variable(tf.zeros([16]), name ="bconv3")
        conv3 = tf.nn.bias_add(conv3, bconv3)
        conv3 = tf.nn.relu(tf.layers.batch_normalization(conv3,momentum=0.9,training =isTrain,epsilon=0.0001))#, name="g_bn_conv3"))
        print(conv3.get_shape(),"conv3")

        #wconv4 = tf.Variable(tf.truncated_normal([30,8,16]), name ="wconv4")
        conv4 = tf.contrib.nn.conv1d_transpose(conv3, wconv4, [batch_size, 27945,8], stride=3, padding="SAME")
        #bconv4 = tf.Variable(tf.zeros([8]), name ="bconv4")
        conv4 = tf.nn.bias_add(conv4, bconv4)
        conv4 = tf.nn.relu(tf.layers.batch_normalization(conv4,momentum=0.9,training =isTrain,epsilon=0.0001))#, name="g_bn_conv4"))
        print(conv4.get_shape(),"conv4")

        #wconv5 = tf.Variable(tf.truncated_normal([54,1,8]), name ="wconv5")
        conv5 = tf.contrib.nn.conv1d_transpose(conv4, wconv5, [batch_size,27998,1], stride=1, padding="VALID")
        #bconv5 = tf.Variable(tf.zeros([1]), name ="bconv5")
        conv5 = tf.nn.bias_add(conv5, bconv5)
        conv5 = tf.layers.batch_normalization(conv5,momentum=0.9,training =isTrain,epsilon=0.0001)
        out = bconv5
        conv5 = tf.nn.tanh(conv5)
        print(conv5.get_shape(),"conv5")
        #conv5 = tf.squeeze(conv5, axis = 2)
        #conv5 = tf.squeeze(conv5, axis = 2)
        return conv5

def discriminator(x,isTrain=True, reuse=False):
    with tf.variable_scope('Discriminator', reuse=reuse):
        
        #print(x.get_shape(),'i')
        #conv1 = conv1d()
        conv1 = tf.layers.conv1d(x,4, 30,strides=2, padding = "Same",kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d())
        conv1 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv1, momentum=0.9, training =isTrain, epsilon=0.00001))
        #conv1 = tf.nn.leaky_relu(conv1)
        conv2 = tf.layers.conv1d(conv1, 16, 30,strides =2,padding = "Same", kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d())
        conv2 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv2, momentum=0.9, training =isTrain, epsilon=0.00001))
        conv3 = tf.layers.conv1d(conv2, 32, 30,strides =2, padding = "Same",kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d())
        conv3 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv3, momentum=0.9, training =isTrain, epsilon=0.00001))
        conv4 = tf.layers.conv1d(conv3, 64, 30,strides =2, padding = "Same",kernel_initializer =tf.contrib.layers.xavier_initializer_conv2d())
        conv4 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv4, momentum=0.9, training =isTrain, epsilon=0.00001))
        print(conv4.get_shape(),"conv4")
        #conv5 = tf.layers.conv1d(conv4, 1,1167, strides = 1,padding ="valid",kernel_initializer =tf.contrib.layers.xavier_initializer_conv2d())
        #print(conv5.get_shape(), "conv5")
        #out = tf.nn.sigmoid(conv5)
        d3 = conv4
        flatten = tf.contrib.layers.flatten(conv4)
        dense1 = tf.layers.dense(flatten, 2048,kernel_initializer = tf.contrib.layers.xavier_initializer())
        #out = tf.nn.sigmoid(dense1)

        dense1 = tf.layers.batch_normalization(dense1, momentum=0.9, training =isTrain, epsilon = 0.00001)#, trainable =False)
        dense1 = tf.nn.leaky_relu(dense1)
        dense2 = tf.layers.dense(dense1, 512,kernel_initializer = tf.contrib.layers.xavier_initializer())
        dense2 = tf.layers.batch_normalization(dense2, momentum=0.9, training =isTrain, epsilon =0.00001)#,trainable =False)
        #d3 = dense2
        dense2 = tf.nn.leaky_relu(dense2)
        dense3 = tf.layers.dense(dense2, 1,kernel_initializer = tf.contrib.layers.xavier_initializer())
        #out = tf.nn.sigmoid(dense3)
        #dense3 = tf.contrib.layers.batch_norm(dense3)
        out = tf.nn.sigmoid(dense3)
        #dense3 = tf.nn.sigmoid(dense3)
        #return dense3
        
        #return out,dense3
        return out,dense3
        """
        conv1 = tf.layers.conv1d(x,8, 30, padding = "Same",kernel_initializer = tf.contrib.layers.xavier_initializer())
        conv1 = tf.nn.tanh(conv1)
        conv1_pool = tf.layers.average_pooling1d(conv1, 3, 3,  padding = "Same")
        conv2 = tf.layers.conv1d(conv1_pool, 16, 30,  padding = "Same", kernel_initializer = tf.contrib.layers.xavier_initializer())
        conv2 = tf.nn.tanh(conv2)
        conv2_pool = tf.layers.average_pooling1d(conv2, 3, 3,  padding = "Same")
        conv3 = tf.layers.conv1d(conv2_pool, 32, 30, padding = "Same",kernel_initializer = tf.contrib.layers.xavier_initializer())
        conv3 = tf.nn.tanh(conv3)
        conv3_pool = tf.layers.average_pooling1d(conv3, 3, 3,  padding = "Same")
        conv4 = tf.layers.conv1d(conv3_pool, 64, 30, padding = "Same",kernel_initializer =tf.contrib.layers.xavier_initializer())
        conv4 = tf.nn.tanh(conv4)
        conv4_pool = tf.layers.average_pooling1d(conv4, 3, 3,  padding = "Same")
        flatten = tf.contrib.layers.flatten(conv4_pool)
        dense1 = tf.layers.dense(flatten, 2048,kernel_initializer = tf.contrib.layers.xavier_initializer())
        dense1 = tf.nn.tanh(dense1)
        dense2 = tf.layers.dense(dense1, 512,kernel_initializer = tf.contrib.layers.xavier_initializer())
        dense2 = tf.nn.tanh(dense2)
        dense3 = tf.layers.dense(dense2, 2,kernel_initializer = tf.contrib.layers.xavier_initializer())
        #dense3 = tf.nn.tanh(dense3)
        dense3 = tf.nn.sigmoid(dense3)
        return dense3
        """
        
        print(x.get_shape(),"xshape")
        print(conv1.get_shape(),'conv1')
        #print(conv1_pool.get_shape(), "pool1")
        print(conv2.get_shape(),'conv2')
        #print(conv2_pool.get_shape(), "pool2")
        print(conv3.get_shape(),'conv3')
        #print(conv3_pool.get_shape(), "pool3")
        print(conv4.get_shape(),'conv4')
        #print(conv4_pool.get_shape(), "pool4")
        print(flatten.get_shape(), "flatten")
        print(dense1.get_shape(), 'dense1')
        #print(dense2.get_shape(), 'dense2')
        #print(dense3.get_shape(), 'dense3')
        #return out, dense3



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


#return dense3

random_vector = tf.placeholder(tf.float32,shape=[None,vector_dim])
real_image_input = tf.placeholder(tf.float32, shape=[None,27998, 1])


isTrain = tf.placeholder(dtype=tf.bool)

gen_sample = generator(random_vector, isTrain, batch_size=batch_size)

disc_real,disc_real_logits = discriminator(real_image_input,isTrain)
disc_fake,disc_fake_logits = discriminator(gen_sample,isTrain, reuse=True)

#disc_real = discriminator(real_image_input,isTrain)
#gan_model = discriminator(gen_sample,isTrain, reuse=True)

#disc_real = discriminator(real_image_input, isTrain)
#disc_fake = discriminator(gen_sample,isTrain, reuse=True)
#disc_concat = tf.concat([disc_real,disc_fake],axis=0)
#print(disc_concat.get_shape(),'concat')

gan_model = discriminator(gen_sample,reuse=True)

#gen_target = tf.placeholder(tf.float32, shape=[None,2])
#disc_target = tf.placeholder(tf.float32, shape=[None,2])
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

optimizer_gen = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)
optimizer_disc = tf.train.AdamOptimizer(learning_rate=0.0002, beta1= 0.5)


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
with tf.Session() as sess:
    #start = time.time()
    sess.run(init)
    #saver.restore(sess, "/home/jack/caltech_research/cell_data_GAN_network/cell_data_GAN_trained.ckpt")
    for i in range(1, num_steps+1):
        if i% 50 == 0:# and i!=500:
            #print("New Data")
            #data = gen_fake_data()
             #data = perfect_data + noise_array()
             start = time.time()
             data = np.apply_along_axis(sin, 1, perfect_data) + noise_array()
             #data = perfect_data+noise_array()
             print(time.time()-start, "time")
        sample = np.random.randint(n_samp, size=batch_size)
        epoch_x = data[sample,:]
        epoch_x = np.reshape(epoch_x, newshape=[-1, 27998, 1])
        z = np.random.uniform(-1.0, 1.0, size=[batch_size, vector_dim])
        #dl,_,dlr,dlf, gl = sess.run([disc_loss,train_op,disc_loss_real, disc_loss_fake,gen_loss], feed_dict = {real_image_input:epoch_x, random_vector:z, isTrain:True})
       	dl,_,dlr,dlf = sess.run([disc_loss,train_disc,disc_loss_real, disc_loss_fake], feed_dict = {real_image_input:epoch_x, random_vector:z, isTrain:True})
        gl, _ = sess.run([gen_loss,train_gen], feed_dict = {random_vector:z,isTrain:True})
        """
        if i == 250:
           real_data = perfect_data + noise_array()
           epoch = real_data[sample,:]
           epoch = np.reshape(epoch, newshape=[-1, 27998, 1])
           out_disc = sess.run([disc_real], feed_dict = {real_image_input:epoch})
           np.save("disc_out.npy", out_disc)
         """
        if dlf < 0.0001:
            save_path = saver.save(sess, "/home/jack/caltech_research/cell_data_GAN_network/cell_data_GAN_trained.ckpt")
            break

        if dl <0.00001:
           save_path = saver.save(sess, "/home/jack/caltech_research/cell_data_GAN_network/cell_data_GAN_trained.ckpt")
           print("done")
           break
           #real_data = perfect_data + noise_array()
           epoch = real_data[sample,:]
           epoch = np.reshape(epoch, newshape=[-1, 27998, 1])
           out_disc = sess.run([disc_real], feed_dict = {real_image_input:epoch})
           np.save("disc_out_0loss.npy", out_disc)
        #if dl<0.00001:
        #   break
        #feed_dict = {real_image_input: epoch_x, random_vector: z,
        #             disc_target: batch_disc_y, gen_target: batch_gen_y, isTrain:True}
        #_, _, gl, dl = sess.run([train_gen, train_disc, gen_loss, disc_loss],
        #                        feed_dict=feed_dict)
        if i % 100 == 0 or i == 1:
            print('Step %i: Generator Loss: %f, Discriminator Loss: %f' % (i, gl, dl))
            print("DLR:",dlr,",", "DLF:", dlf)
    save_path = saver.save(sess, "/home/jack/caltech_research/cell_data_GAN_network/cell_data_GAN_trained.ckpt")

    sample = np.random.randint(n_samp, size=batch_size)
    epoch_x = data[sample,:]
    epoch_x = np.reshape(epoch_x, newshape=[-1, 27998,1])
    print(epoch_x)
    z = np.random.uniform(-1., 1., size=[batch_size, vector_dim])
    #gen = generator(random_vector,batch_size=100,reuse=True)
    samp  = sess.run(gen_sample, feed_dict={random_vector: z, isTrain:False})
    
    g = sess.run(disc_real, feed_dict={real_image_input: epoch_x, isTrain:False})
    o = sess.run(disc_real, feed_dict={real_image_input: samp, isTrain:False})
    #o = epoch_x
    np.save("disc_output_fake.npy",o)    
    np.save("disc_output.npy", g)
    np.save("generator_output.npy", samp)
    #np.save("generator_output2.npy", samp2)
