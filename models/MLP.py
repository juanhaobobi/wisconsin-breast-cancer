# Copyright 2017 Abien Fred Agarap
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

"""Implementation of the Multilayer Perceptron using TensorFlow"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = "0.1.0"
__author__ = "Abien Fred Agarap"

import numpy as np
import os
import sys
import time
import tensorflow as tf


import numpy as np
import os
import tensorflow as tf
import time

class MLP:
    def __init__(self, alpha, batch_size, node_size, num_classes, num_features):
        self.alpha = alpha
        self.batch_size = batch_size
        self.node_size = node_size
        self.num_classes = num_classes
        self.num_features = num_features

        self.optimizer = tf.optimizers.SGD(learning_rate=self.alpha)
        self.train_loss = tf.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.metrics.CategoricalAccuracy(name='train_accuracy')
        self.test_loss = tf.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.metrics.CategoricalAccuracy(name='test_accuracy')

        self.build_model()

    def build_model(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(self.node_size[0], activation='relu', input_shape=(self.num_features,)),
            tf.keras.layers.Dense(self.node_size[1], activation='relu'),
            tf.keras.layers.Dense(self.node_size[2], activation='relu'),
            tf.keras.layers.Dense(self.num_classes)
        ])

    @tf.function
    def train_step(self, features, labels):
        with tf.GradientTape() as tape:
            predictions = self.model(features, training=True)
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=predictions))
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.train_loss(loss)
        self.train_accuracy(labels, tf.nn.softmax(predictions))

    @tf.function
    def test_step(self, features, labels):
        predictions = self.model(features, training=False)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=predictions))
        self.test_loss(loss)
        self.test_accuracy(labels, tf.nn.softmax(predictions))

    def train(self, num_epochs, train_data, test_data, log_dir, result_path):
        train_dataset = tf.data.Dataset.from_tensor_slices((train_data[0], train_data[1])).shuffle(buffer_size=10000).batch(self.batch_size)
        test_dataset = tf.data.Dataset.from_tensor_slices((test_data[0], test_data[1])).batch(self.batch_size)

        current_time = time.strftime("%Y%m%d-%H%M%S")
        train_log_dir = os.path.join(log_dir, current_time, 'train')
        test_log_dir = os.path.join(log_dir, current_time, 'test')
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)

        for epoch in range(num_epochs):
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.test_loss.reset_states()
            self.test_accuracy.reset_states()

            for features, labels in train_dataset:
                self.train_step(features, labels)

            with train_summary_writer.as_default():
                tf.summary.scalar('loss', self.train_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', self.train_accuracy.result(), step=epoch)

            for features, labels in test_dataset:
                self.test_step(features, labels)

            with test_summary_writer.as_default():
                tf.summary.scalar('loss', self.test_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', self.test_accuracy.result(), step=epoch)

            print(f'Epoch {epoch + 1}, '
                  f'Loss: {self.train_loss.result():.4f}, '
                  f'Accuracy: {self.train_accuracy.result() * 100:.2f}%, '
                  f'Test Loss: {self.test_loss.result():.4f}, '
                  f'Test Accuracy: {self.test_accuracy.result() * 100:.2f}%')

        # Save the model
        self.model.save('mlp_model.h5')

        # Save labels
        predictions = []
        actual = []
        for features, labels in test_dataset:
            predictions.extend(np.argmax(self.model(features, training=False), axis=1))
            actual.extend(labels.numpy())

        labels = np.column_stack((predictions, actual))
        np.save(os.path.join(result_path, "mlp_labels.npy"), labels)
