'''
    ------------------------------------
    Author : SAHLI Mohammed
    Date   : 2019-11-13
    Company: Alphaya (www.alphaya.com)
    Email  : nihon.sahli@gmail.com
    ------------------------------------
'''

# Date : 2016-01
# Paper: Unsupervised Representation Learning with Deep Convolutional Generative
#        Adversarial Networks
# Link : https://arxiv.org/abs/1511.06434

import os
import cv2
import sys
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
sys.path.append('..')
import utils.loader as loader
import utils.utils  as utils
import utils.layers as layers

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Adding Seed so that random initialization is consistent
# from tensorflow import set_random_seed
# np.random.seed(1)
# set_random_seed(2)

class DCGAN:
    #................................................................................
    # Constructor
    #................................................................................
    def __init__(self, model_path, data_path = None, is_training = False,
                 batch_size = 32, image_size = 28, latent_dim = 128,
                 hard_load = False, verbose = False, pretrained = False):
        self.pretrained   = pretrained
        self.image_size   = image_size  # height and weight of images
        self.channels     = 3           # dealing with rgb images, hence 3 channels
        self.batch_size   = batch_size
        self.latent_dim   = latent_dim
        self.linear_dim   = 1024
        self.conv_dim     = 128
        self.g_lr_rate    = 0.0002 * 5
        self.d_lr_rate    = 0.0002
        self.beta1        = 0.5
        self.w_initializer= "xavier"    # or "uniform", "gaussian", "truncated"
        self.b_initializer= "constant"  # or "normal"

        self.verbose      = verbose
        self.model_path   = model_path
        self.train_path   = data_path
        self.session      = tf.Session()

        # DCGAN Parameters
        self.model_name   = "DCGAN"
        self.gf_dim       = 64
        # self.df_dim     = 64
        # self.gfc_dim    = 1024
        # self.dfc_dim    = 1024
        # self.c_dim      = 3

        if is_training == True:
            self.train_initialization(hard_load     = hard_load,
                                      di_iterations = 1,
                                      ge_iterations = 1)
        else:
            self.predict_initialization()

    #................................................................................
    # Deconstructor
    #................................................................................
    def __del__(self):
        pass

    #................................................................................
    # Training Initialization
    #................................................................................
    def train_initialization(self, hard_load = True, di_iterations = 1, ge_iterations = 1):
        # initialize training dataset
        self.data = loader.DataSet(images_dir = self.train_path,
                                   width      = self.image_size,
                                   height     = self.image_size,
                                   hard_load  = hard_load,
                                   verbose    = self.verbose    )

        self.di_iterations = max(1, di_iterations)
        self.ge_iterations = max(1, ge_iterations)

        # get number of batches for a single epoch
        self.num_batches = self.data.size // self.batch_size

        if self.verbose:
            print("Data size  =", self.data.size )
            print("batch_size =", self.batch_size)
            print("ge_lr_rate =", self.g_lr_rate )
            print("di_lr_rate =", self.d_lr_rate )

        # Network Inputs
        g_input_shape= [self.batch_size, self.latent_dim]
        d_input_shape= [self.batch_size, self.image_size, self.image_size, self.channels]
        self.g_input = tf.placeholder(tf.float32, shape = g_input_shape, name='ge_input')
        self.d_input = tf.placeholder(tf.float32, shape = d_input_shape, name='di_input')

        # create the network's model and optimizer
        self.create_network()
        self.create_optimizer()

        # initialize of all global variables
        global_variables = tf.global_variables_initializer()
        self.session.run(global_variables)
        self.saver = tf.train.Saver()

        if self.pretrained == True:
            if self.verbose == True: print("Loading pretrained model...",end='')
            meta_graph = self.model_path + self.model_name + '.meta'
            checkpoint = tf.train.latest_checkpoint(self.model_path)  #
            self.saver.restore(self.session, checkpoint)              # Load the weights
            if self.verbose == True: print("done")

    #................................................................................
    # Prediction Initialization
    #................................................................................
    def predict_initialization(self):
        meta_graph = self.model_path + self.model_name + '.meta'
        self.saver = tf.train.import_meta_graph(meta_graph)       # Recreate the network graph
        checkpoint = tf.train.latest_checkpoint(self.model_path)  #
        self.saver.restore(self.session, checkpoint)              # Load the weights
        self.graph = tf.get_default_graph()

    #................................................................................
    # DCGAN Discriminator Architecture
    #................................................................................
    def discriminator(self, x, is_training=True, reuse=False):
        with tf.variable_scope("discriminator", reuse = reuse):
            if self.verbose: print(x.shape)

            # Layer 1
            net = layers.conv2d(x, self.conv_dim >> 1, name='di_conv1')
            net = tf.nn.leaky_relu(net)
            if self.verbose: print(net.shape)

            # Layer 2
            net = layers.conv2d(net, self.conv_dim >> 0, name='di_conv2')
            net = layers.batch_norm(net, is_training = is_training, scope='di_bn1')
            net = tf.nn.leaky_relu(net)
            if self.verbose: print(net.shape)

            # Layer 3
            net = layers.conv2d(net, self.conv_dim << 1, name='di_conv3')
            net = layers.batch_norm(net, is_training = is_training, scope='di_bn2')
            net = tf.nn.leaky_relu(net)
            if self.verbose: print(net.shape)

            # Layer 4
            net = layers.conv2d(net, self.conv_dim << 2, name='di_conv4')
            net = layers.batch_norm(net, is_training = is_training, scope='di_bn3')
            net = tf.nn.leaky_relu(net)
            if self.verbose: print(net.shape)

            # Layer 5
            net = tf.reshape(net, [self.batch_size, -1])
            if self.verbose: print(net.shape)

            # Layer 6
            out_logit = layers.linear(net, 1, scope='di_fc1')
            out= tf.nn.sigmoid(out_logit)
            if self.verbose: print(out.shape)

            return out, out_logit

    #................................................................................
    # DCGAN Generator Architecture
    #................................................................................
    def generator(self, z, is_training=True, reuse=False):
        with tf.variable_scope("generator", reuse=reuse):
            if self.verbose: print(z.shape)
            h = self.image_size
            w = self.image_size

            # Layer 1
            net = layers.linear(z, (self.gf_dim << 3) * (h >> 4) * (w >> 4), 'ge_fc1')
            net = tf.reshape(net, [-1, (h >> 4), (w >> 4), self.gf_dim << 3])
            net = layers.batch_norm(net, is_training=is_training,scope='ge_bn1')
            net = tf.nn.relu(net)
            if self.verbose: print(net.shape)

            # Layer 2
            shape = [self.batch_size, h >> 3, w >> 3, self.gf_dim << 2]
            net = layers.deconv2d(net, shape, name = 'ge_dc1')
            net = layers.batch_norm(net, is_training=is_training,scope='ge_bn2')
            net = tf.nn.relu(net)
            if self.verbose: print(net.shape)

            # Layer 3
            shape = [self.batch_size, h >> 2, w >> 2, self.gf_dim << 1]
            net = layers.deconv2d(net, shape, name = 'ge_dc2')
            net = layers.batch_norm(net, is_training=is_training,scope='ge_bn3')
            net = tf.nn.relu(net)
            if self.verbose: print(net.shape)

            # Layer 4
            shape = [self.batch_size, h >> 1, w >> 1, self.gf_dim << 0]
            net = layers.deconv2d(net, shape, name = 'ge_dc3')
            net = layers.batch_norm(net, is_training=is_training,scope='ge_bn4')
            net = tf.nn.relu(net)
            if self.verbose: print(net.shape)

            # Layer 5
            shape = [self.batch_size, h, w, self.channels]
            h4 = layers.deconv2d(net, shape, name = 'ge_dc4')
            out = tf.nn.sigmoid(h4, name = "main_out")
            if self.verbose: print(out.shape)

            return out

    #................................................................................
    # Construct the network
    #................................................................................
    def create_network(self):
        # output of D for real images
        outs = self.discriminator(self.d_input, is_training=True, reuse=False)
        self.d_real        = outs[0]
        self.d_real_logits = outs[1]

        # output of D for fake images
        if self.verbose: print("Generator")
        self.g_output = self.generator(self.g_input, is_training=True, reuse=False)

        if self.verbose: print("Discriminator")
        outs = self.discriminator(self.g_output, is_training=True, reuse=True)
        self.d_fake        = outs[0]
        self.d_fake_logits = outs[1]

    #................................................................................
    # Create the Optimizer
    #................................................................................
    def create_optimizer(self):
        # Discrimminator & Generator Losses
        # Paper, p3: log(1 − D(G(z))) saturates. Rather than training G to minimize
        #            log(1 − D(G(z))) we can train G to maximize log D(G(z))
        eps = 1e-04
        self.ge_loss = -tf.reduce_mean(tf.log(eps + self.d_fake))
        self.di_loss = -tf.reduce_mean(tf.log(eps + self.d_real)) - tf.reduce_mean(tf.log(eps + 1.0 - self.d_fake))
        # self.ge_loss = tf.reduce_mean(tf.log(eps + 1.0 - self.d_fake)) # See note above

        # Another alternative
        # d_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_real_logits, labels=tf.ones_like(self.d_real)))
        # d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake_logits, labels=tf.zeros_like(self.d_fake))
        # self.d_loss = tf.reduce_mean(d_loss_real) + tf.reduce_mean(d_loss_fake)
        # g_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake_logits, labels=tf.ones_like(self.d_fake))
        # self.g_loss = tf.reduce_mean(g_loss_fake)

        # Optimizers
        g_optimizer = tf.train.AdamOptimizer(self.g_lr_rate, beta1=self.beta1)
        d_optimizer = tf.train.AdamOptimizer(self.d_lr_rate, beta1=self.beta1)

        # Training Variables for each optimizer
        # By default in TensorFlow, all variables are updated by each optimizer, so
        # we need to precise for each one of them the specific variables to update.
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'di_' in var.name]
        g_vars = [var for var in t_vars if 'ge_' in var.name]

        # Create training operations
        self.ge_opt = g_optimizer.minimize(self.ge_loss, var_list = g_vars)
        self.di_opt = d_optimizer.minimize(self.di_loss, var_list = d_vars)

    #................................................................................
    # Training Function
    #................................................................................
    def train(self, max_epoches, show_images = False):
        # initialize of all global variables
        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

        # start training
        for epoch in range(max_epoches):
            for i in range(self.num_batches):
                x_batch, _ = self.data.next_batch(self.batch_size)
                # z_batch = np.random.uniform(-1.0, 1.0, size = [self.batch_size, self.latent_dim])
                z_batch = np.random.normal(0.0, 1.0, size = [self.batch_size, self.latent_dim])

                for _ in range(self.di_iterations):
                    _, di_loss = self.session.run([self.di_opt, self.di_loss],
                                                   feed_dict = {self.d_input: x_batch,
                                                                self.g_input: z_batch})

                for _ in range(self.ge_iterations):
                    _, ge_loss = self.session.run([self.ge_opt, self.ge_loss],
                                                   feed_dict = {self.d_input: x_batch,
                                                                self.g_input: z_batch})
                if i % 100 == 0:
                    msg = "Epoch {}-{}/{}\tge_loss: {:.5f}\tdi_loss: {:.5f}"
                    print(msg.format(epoch + 1, i + 1, self.num_batches, ge_loss, di_loss))
                    if math.isnan(ge_loss) == False and math.isnan(di_loss) == False:
                        self.saver.save(self.session, self.model_path + self.model_name)
                        print("recent model was saved to",self.model_path + self.model_name)
                    else:
                        sys.exit()

                    if show_images == True:
                        plt.close()
                        f, a = plt.subplots(4, 10, figsize=(10, 4))
                        for k in range(10):
                            # z = np.random.uniform(-1.0, 1.0, size = [self.batch_size, self.latent_dim])
                            z = np.random.normal(0.0, 1.0, size = [self.batch_size, self.latent_dim])
                            g = self.session.run(self.g_output, feed_dict = {self.g_input: z})
                            for j in range(4):
                                a[j][k].imshow(g[j])

                        f.suptitle("Epoch "+str(epoch)+", Step "+str(i), fontsize=9)
                        f.show()
                        plt.draw()
                        plt.pause(0.001)

            z = np.random.normal(0.0, 1.0, size = [self.batch_size, self.latent_dim])
            g = self.session.run(self.g_output, feed_dict = {self.g_input: z})
            grid = self.construct_image_grid(g, 30, 480, 240)
            cv2.imwrite("images/"+self.model_name+"/grid_" + str(epoch+1) + ".png", grid)

        if show_images:
            plt.close()

    #................................................................................
    #
    #................................................................................
    def construct_image_grid(self, preds, samples, grid_width, grid_height):
        preds = preds[:samples]
        preds = np.reshape(preds,(3,-1,self.image_size,self.image_size,self.channels))
        grid = [np.concatenate(tuple(preds[0]), axis = 1),
                np.concatenate(tuple(preds[1]), axis = 1),
                np.concatenate(tuple(preds[2]), axis = 1)]
        grid = np.concatenate(tuple(grid), axis = 0)
        grid = cv2.resize(grid, (grid_width, grid_height),
                          interpolation = cv2.INTER_AREA)
        grid  = (grid * 255.0).astype(np.uint8)
        return grid

    #................................................................................
    #
    #................................................................................
    def generate(self, destination = None, samples     = 30,
                       grid_width  = 480 , grid_height = 240):
        input  = self.graph.get_tensor_by_name("ge_input:0")
        output = self.graph.get_tensor_by_name("generator/main_out:0")
        # z    = np.random.uniform(-1.0, 1.0, size = [self.batch_size, self.latent_dim])
        z      = np.random.normal(0.0, 1.0, size = [self.batch_size, self.latent_dim])
        preds  = self.session.run(output, feed_dict = {input: z})
        # preds  = (preds * 255.0).astype(np.uint8)
        grid   = self.construct_image_grid(preds, samples, grid_width, grid_height)
        cv2.imwrite(destination, grid)
