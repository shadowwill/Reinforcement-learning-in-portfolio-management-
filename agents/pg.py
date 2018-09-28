#-*- coding:utf-8 -*-
'''
@Author: Louis Liang
@time:2018/9/15 0:34
'''
import tensorflow as tf
import tflearn
import numpy as np
import util
import math

class PG:
    def __init__(self,M,L,N,name,load_weights,trainable):
        # Initial buffer
        self.buffer = list()
        self.name = name
        self.learning_rate=10e-4

        # Build up models
        self.sesson = tf.Session()

        # Initial input shape
        self.M = M
        self.L = L
        self.N = N
        self.global_step = tf.Variable(0, trainable=False)

        self.index_weights = tf.constant(np.ones(self.M) / self.M, dtype=tf.float32)
        self.state,self.w_previous,self.out, =self.build_net()  #self.out:最新权重
        self.future_price=tf.placeholder(tf.float32,[None]+[self.M])
        self.pv_vector=tf.reduce_sum(self.out*self.future_price,reduction_indices=[1])*self.pc() ##self.pc():交易成本
        self.profit=tf.reduce_prod(self.pv_vector)
        self.loss=-tf.reduce_mean(tf.log(self.pv_vector))
        #self.loss=-tf.reduce_mean(tf.log(self.pv_vector))/util.reduce_std(tf.log(self.pv_vector))+ 0.11 * self.tracking_error()

        self.maxbias = self.max_bias()
        #self.loss = tf.cond(self.maxbias > tf.constant(0.055), lambda: -tf.reduce_mean(tf.log(self.pv_vector)) / util.reduce_std(tf.log(self.pv_vector))+10, lambda:-tf.reduce_mean(tf.log(self.pv_vector)) / util.reduce_std(tf.log(self.pv_vector)))
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

            # Initial saver
        self.saver = tf.train.Saver(max_to_keep=10)
        if load_weights == 'True':
            print("Loading Model")
            try:
                checkpoint = tf.train.get_checkpoint_state('./saved_network/PG')
                if checkpoint and checkpoint.model_checkpoint_path:
                    self.saver.restore(self.sesson, checkpoint.model_checkpoint_path)
                    print("Successfully loaded:", checkpoint.model_checkpoint_path)
                else:
                    print("Could not find old network weights")
                    self.sesson.run(tf.global_variables_initializer())
            except:
                print("Could not find old network weights")
                self.sesson.run(tf.global_variables_initializer())
        else:
            self.sesson.run(tf.global_variables_initializer())

        if trainable == 'True':
            # Initial summary
            self.summary_writer = tf.summary.FileWriter('./summary/PG', self.sesson.graph)
            #self.summary_ops, self.summary_vars = build_summaries()


    # 建立 policy gradient 神经网络 (有改变)
    def build_net(self):
        state=tf.placeholder(tf.float32,shape=[None]+[self.M]+[self.L]+[self.N],name='market_situation')
        network = tflearn.layers.conv_2d(state, 2,
                                         [1, 2],
                                         [1, 1, 1, 1],
                                         'valid',
                                         'relu')
        width = network.get_shape()[2]
        network = tflearn.layers.conv_2d(network, 48,
                                         [1, width],
                                         [1, 1],
                                         "valid",
                                         'relu',
                                         regularizer="L2",
                                         weight_decay=5e-9)
        w_previous=tf.placeholder(tf.float32,shape=[None,self.M])
        network=tf.concat([network,tf.reshape(w_previous, [-1, self.M, 1, 1])],axis=3)
        network = tflearn.layers.conv_2d(network, 1,
                                         [1, network.get_shape()[2]],
                                         [1, 1],
                                         "valid",
                                         'relu',
                                         regularizer="L2",
                                         weight_decay=5e-9)
        network=tf.layers.flatten(network)
        w_init = tf.random_uniform_initializer(-0.003, 0.003)
        out = tf.layers.dense(network, self.M, activation=tf.nn.softmax, kernel_initializer=w_init)
        


        return state,w_previous,out

    def max_bias(self):
        return tf.reduce_max(tf.abs(self.out[:,:]-self.index_weights[:]))

    def tracking_error(self):
        return tf.reduce_sum(tf.abs(self.out[:,:]-self.index_weights[:]),axis=1)

    def pc(self):
        return 1-tf.reduce_sum(tf.abs(self.out[:,1:]-self.w_previous[:,1:]),axis=1)*0.0025

    # 选行为 (有改变)
    def predict(self, s, a_previous):
        index_weight = np.array([[1 / self.M for i in range(self.M)]])
        w = self.sesson.run(self.out,feed_dict={self.state:s,self.w_previous:a_previous})
        return w# np.array([ index_weight[0][:] - (w[0][:] -  index_weight[0][:]) ])

    # 存储回合 transition (有改变)
    def save_transition(self, s, p, action,action_previous):
        self.buffer.append((s, p, action,action_previous))

    # 学习更新参数 (有改变)
    def train(self):
        s,p,a,a_previous=self.get_buffer()
        maxbias,loss,_=self.sesson.run([self.maxbias,self.loss,self.optimize],feed_dict={self.state:s,
                                                                        self.out:np.reshape(a,(-1,self.M)),
                                                                        self.future_price:np.reshape(p,(-1,self.M)),
                                                                        self.w_previous:np.reshape(a_previous,(-1,self.M))})
        print(maxbias)
        if maxbias > 0.1:
            import sys
            sys.exit()

        #print(loss)
        self.save_model()

    def get_buffer(self):
        s = [data[0][0] for data in self.buffer]
        p = [data[1] for data in self.buffer]
        a = [data[2] for data in self.buffer]
        a_previous = [data[3] for data in self.buffer]
        return s, p,a,a_previous

    def reset_buffer(self):
        self.buffer = list()

    def save_model(self):
        self.saver.save(self.sesson,'./saved_network/PG/'+self.name,global_step=self.global_step)