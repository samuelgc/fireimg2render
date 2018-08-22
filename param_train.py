from subprocess import call
from PIL import Image

import tensorflow as tf
import numpy as np
import random
import csv
import os


class ParamLearner:
    input_size = [None, 256, 256, 3]
    output_size = [None, 10]
    model_directory = '.\models\param_learner_model'

    def __init__(self):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.input = tf.placeholder(tf.float32, self.input_size, name='input')
        self.target = tf.placeholder(tf.float32, self.output_size, name='target')

        #Convolutional layers
        conv0 = tf.layers.conv2d(self.input, 64, 3, padding="same", activation=tf.nn.relu, name='conv0')
        pool0 = tf.layers.max_pooling2d(conv0, 2, 2, padding="same", name='pool0')
        conv1 = tf.layers.conv2d(pool0, 32, 3, padding="same", activation=tf.nn.relu, name='conv1')
        pool1 = tf.layers.max_pooling2d(conv1, 2, 2, padding="same", name='pool1')
        conv2 = tf.layers.conv2d(pool1, 16, 3, padding="same", activation=tf.nn.relu, name='conv2')
        pool2 = tf.layers.max_pooling2d(conv2, 2, 2, padding="same", name='pool2')

        #Dense layers
        flat = tf.reshape(pool2, [-1])
        dense0 = tf.layers.dense(flat, units=1024, activation=tf.nn.relu, name='dense0')
        dense1 = tf.layers.dense(dense0, units=512, activation=tf.nn.relu, name='dense1')
        self.output = tf.layers.dense(dense1, units=10, activation=tf.nn.sigmoid, name='output')

        self.loss = tf.reduce_sum(tf.abs(self.target - self.output), name='loss')
        self.cost = tf.reduce_mean(self.loss)
        self.train = tf.train.AdamOptimizer().minimize(self.cost, name='train')

        tf.add_to_collection('input', self.input)
        tf.add_to_collection('target', self.target)
        tf.add_to_collection('output', self.output)
        tf.add_to_collection('loss', self.loss)
        tf.add_to_collection('cost', self.cost)
        tf.add_to_collection('train', self.train)
        self.saver = tf.train.Saver()

    def train(self, input_data, load_existing_model=True):
        self.sess.run(tf.global_variables_initializer())
        if load_existing_model and os.path.exists(self.model_directory):
            self.load_trained_model()
        else:
            self.sess.run(tf.global_variables_initializer())

        for epoch in range(100):
            count = 0
            try:
                params = input_data[count]
                if epoch == 0:
                    with open('./ifds/fire.ifd') as f:
                        search_string = "fc_colorramp_the_basis_strings ( \"linear\" \"linear\" ) fc_colorramp_the_key_positions ( 0 1 ) fc_colorramp_the_key_values ( 0 0 0 1 1 1 )"
                        replace_string = "s_densityscale {} s_int {} s_color {} {} {} fi_int {} fc_int {} fc_colorramp_the_basis_strings ( \"linear\" \"linear\" ) fc_colorramp_the_key_positions ( 0 1 ) fc_colorramp_the_key_values ( 0 0 0 1 1 1 ) fc_bbtemp {} fc_bbadapt {} fc_bbburn {}" \
                            .format(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9])
                        contents = f.read().replace(search_string, replace_string)
                    with open('./ifds/render_fire_{}_{}.ifd'.format(epoch, count), "w+") as f:
                        f.write(contents)
                    call(["mantra", "./ifds/render_fire_{}_{}.ifd".format(epoch, count), "./render/render_{}_{}.jpg".format(count)])
                img = Image.open("./render/render_{}_{}.jpg".format(count))
                img_in = np.asarray(img)
                feed_dict = {self.input: img_in, self.target: params}
                loss, _ = self.sess.run([self.cost, self.output], feed_dict=feed_dict)
                count += 1
            except Exception as fail:
                continue
            finally:
                self.save_trained_model()

    def save_trained_model(self):
        self.saver.save(self.sess, os.path.join(self.model_directory, 'model'))

    def load_trained_model(self):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(self.model_directory))


def generate_samples(size):
    samples = []
    for count in range(size):
        sample = []
        #Generate shader parameters randomly
        sample.append(random.uniform(0, 2))             # Density Scale
        sample.append(random.uniform(0, 2))             # Smoke Brightness
        smoke_color = random.random()                   # Smoke Color
        sample.append(smoke_color)
        sample.append(smoke_color)
        sample.append(smoke_color)
        sample.append(random.uniform(0, 5))             # Intensity Scale
        sample.append(random.uniform(0, 5))             # Temperature Scale
        sample.append(int(random.uniform(0, 15000)))    # Color Temp in Kelvin
        sample.append(random.random())                  # Adaption
        sample.append(random.uniform(-2, 2))            # Burn
        samples.append(sample)
    return samples


def create_training_file(size=100):
    with open('./train_data/shader_params.csv', "w+") as f:
        writer = csv.writer(f)
        writer.writerows(generate_samples(size))
        f.close()


def main():
    param_learner = ParamLearner()
    generate_samples(100)
    data = np.loadtxt('./train_data/shader_params.csv', delimiter=",")
    param_learner.train(input_data=data)


if __name__ == '__main__':
    main()
