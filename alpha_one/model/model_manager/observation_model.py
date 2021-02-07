#!/usr/bin/env python
# coding: utf-8

# In[4]:


import collections
import functools
import os
from typing import Sequence

import numpy as np
import tensorflow.compat.v1 as tf

class TrainInput(collections.namedtuple( "TrainInput", "observation legals_mask policy value")):

    @staticmethod
    def stack(train_inputs):
        observation, legals_mask, policy, value = zip(*train_inputs)
        return TrainInput(
            np.array(observation, dtype=np.float32),
            np.array(legals_mask, dtype=np.bool),
            np.array(policy),
            np.expand_dims(value, 1))

class Model(object):
    
    
    def __init__(self, session, saver, path):
        
        self._session = session
        self._saver = saver
        self._path = path
    
    
        def get_var(name):
            return self._session.graph.get_tensor_by_name(name + ":0")

        self._input = get_var("input")
        self._legals_mask = get_var("legals_mask")
        self._training = get_var("training")
        self._value_out = get_var("value_out")
        self._policy_softmax = get_var("policy_softmax")
        self._policy_loss = get_var("policy_loss")
        self._value_loss = get_var("value_loss")
        self._l2_reg_loss = get_var("l2_reg_loss")
        self._policy_targets = get_var("policy_targets")
        self._value_targets = get_var("value_targets")
        self._train = self._session.graph.get_operation_by_name("train")
    
    
    
    @classmethod
    def build_model(cls, input_shape, output_size, nn_width, nn_depth,
                    weight_decay, learning_rate, path):


        g = tf.Graph() 
        with g.as_default():
            cls._define_graph(input_shape, output_size, nn_width,
                            nn_depth, weight_decay, learning_rate)
            
            init = tf.variables_initializer(tf.global_variables(),
                                          name="init_all_vars_op")
            with tf.device("/cpu:0"):
                saver = tf.train.Saver(
                    max_to_keep=10000, sharded=False, name="saver")
        
        session = tf.Session(graph=g)
        session.__enter__()
        session.run(init)
        return cls(session, saver, path)
    
    
    @staticmethod
    def _define_graph(input_shape, output_size,
                        nn_width, nn_depth, weight_decay, learning_rate):
        
        input_size = int(np.prod(input_shape))
        observations = tf.placeholder(tf.float32, [None, input_size], name="input")
        legals_mask = tf.placeholder(tf.bool, [None, output_size],
                                     name="legals_mask")
        
        training = tf.placeholder(tf.bool, name="training")
        
        bn_updates = []
        
        layer = observations
        
        for i in range(nn_depth):
            layer = tf.keras.layers.Dense(nn_width, name=f"layer_{i}_dense")(layer)
            layer = tf.keras.layers.Activation("relu")(layer)
        
        policy_head = tf.keras.layers.Dense(nn_width, name="policy_dense")(layer)
        
        policy_head = tf.keras.layers.Activation("relu")(policy_head)
        
        policy_logits = tf.keras.layers.Dense(output_size, name="policy")(policy_head)
        
        policy_logits = tf.where(legals_mask, policy_logits, -1e32 * tf.ones_like(policy_logits))
        
        policy_softmax = tf.identity(tf.keras.layers.Softmax()(policy_logits), name="policy_softmax")
        
        policy_targets = tf.placeholder(shape=[None, output_size], dtype=tf.float32, name="policy_targets")
        
        policy_loss = tf.reduce_mean(
                                    tf.nn.softmax_cross_entropy_with_logits_v2(
                                    logits=policy_logits, labels=policy_targets),
                                    name="policy_loss")
        
        
        value_head = layer
        
        value_out = tf.keras.layers.Dense(nn_width, name="value_dense")(value_head)
        
        value_out = tf.keras.layers.Activation("relu")(value_out)
        
        value_out = tf.keras.layers.Dense(1, name="value")(value_out)
        
        value_out = tf.keras.layers.Activation("tanh")(value_out)
        
        value_out = tf.identity(value_out, name="value_out")
        
        value_targets = tf.placeholder(
                                    shape=[None, 1], dtype=tf.float32, name="value_targets")
        
        value_loss = tf.identity(tf.losses.mean_squared_error(
                                value_out, value_targets), name="value_loss")
        
        
        l2_reg_loss = tf.add_n([ weight_decay * tf.nn.l2_loss(var) for var in tf.trainable_variables()
                                                                        if "/bias:" not in var.name
                                                                            ], name="l2_reg_loss")
        
        

        total_loss = policy_loss + value_loss + l2_reg_loss
        
        optimizer = tf.train.AdamOptimizer(learning_rate)
        
        with tf.control_dependencies(bn_updates):
            train = optimizer.minimize(total_loss, name="train")
            
            
    @classmethod
    def from_checkpoint(cls, checkpoint, path=None):
        model = cls.from_graph(checkpoint, path)
        model.load_checkpoint(checkpoint)
        return model

    @classmethod
    def from_graph(cls, metagraph, path=None):
        if not os.path.exists(metagraph):
            metagraph += ".meta"
        if not path:
            path = os.path.dirname(metagraph)
        g = tf.Graph()  # Allow multiple independent models and graphs.
        with g.as_default():
            saver = tf.train.import_meta_graph(metagraph)
        session = tf.Session(graph=g)
        session.__enter__()
        session.run("init_all_vars_op")
        return cls(session, saver, path)
    
    
    @property
    def num_trainable_variables(self):
        return sum(np.prod(v.shape) for v in tf.trainable_variables())

    def print_trainable_variables(self):
        for v in tf.trainable_variables():
            print("{}: {}".format(v.name, v.shape))

    def write_graph(self, filename):
        full_path = os.path.join(self._path, filename)
        tf.train.export_meta_graph(
            graph_def=self._session.graph_def, saver_def=self._saver.saver_def,
            filename=full_path, as_text=False)
        return full_path

    def inference(self, observation, legals_mask):
        return self._session.run(
            [self._value_out, self._policy_softmax],
            feed_dict={self._input: np.array(observation, dtype=np.float32),
                        self._legals_mask: np.array(legals_mask, dtype=np.bool),
                        self._training: False})

    def update(self, train_inputs: Sequence[TrainInput]):
        batch = TrainInput.stack(train_inputs)

        _, policy_loss, value_loss, l2_reg_loss = self._session.run(
                        [self._train, self._policy_loss, self._value_loss, self._l2_reg_loss],
                        feed_dict={self._input: batch.observation,
                                self._legals_mask: batch.legals_mask,
                                self._policy_targets: batch.policy,
                                self._value_targets: batch.value,
                                self._training: True})

        return policy_loss, value_loss, l2_reg_loss

    def save_checkpoint(self, step):
        return self._saver.save(
            self._session,
            os.path.join(self._path, "checkpoint"),
            global_step=step)

    def load_checkpoint(self, path):
        return self._saver.restore(self._session, path)
                  


# In[ ]:





# In[ ]:




