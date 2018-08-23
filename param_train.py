from subprocess import call
from PIL import Image

import tensorflow as tf
import numpy as np
import random
import csv


class ParamLearner:
    input_size = [None, 256, 256, 3]
    output_size = [None, 10]

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
        flat = tf.reshape(pool2, [1, 16384])
        dense0 = tf.layers.dense(flat, units=1024, activation=tf.nn.relu, name='dense0')
        dense1 = tf.layers.dense(dense0, units=512, activation=tf.nn.relu, name='dense1')
        dense2 = tf.layers.dense(dense1, units=256, activation=tf.nn.relu, name='dense2')
        self.output = tf.layers.dense(dense2, units=10, activation=tf.nn.sigmoid, name='output')

        self.loss = tf.reduce_sum(tf.abs(self.target - self.output), name='loss')
        self.train = tf.train.AdamOptimizer().minimize(self.loss, name='train')

    def start_train(self, fresh=True, sample_size=500):
        if fresh:
            create_training_file(sample_size)
            print "New training file generated"
        data = np.loadtxt('./train_data/shader_params.csv', delimiter=",")
        self.sess.run(tf.global_variables_initializer())
        for epoch in range(1000):
            total_loss = 0
            for count in range(len(data)):
                try:
                    params = data[count]
                    if fresh and epoch == 0:
                        with open('./ifds/fire.ifd') as f:
                            search_string = "fc_colorramp_the_basis_strings ( \"linear\" \"linear\" ) fc_colorramp_the_key_positions ( 0 1 ) fc_colorramp_the_key_values ( 0 0 0 1 1 1 )"
                            replace_string = "s_densityscale {} s_int {} s_color {} {} {} fi_int {} fc_int {} fc_colorramp_the_basis_strings ( \"linear\" \"linear\" ) fc_colorramp_the_key_positions ( 0 1 ) fc_colorramp_the_key_values ( 0 0 0 1 1 1 ) fc_bbtemp {} fc_bbadapt {} fc_bbburn {}" \
                                .format(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9])
                            contents = f.read().replace(search_string, replace_string)
                        with open('./ifds/render_fire_{}_{}.ifd'.format(epoch, count), "w+") as f:
                            f.write(contents)
                        call(["mantra", "./ifds/render_fire_{}_{}.ifd".format(epoch, count), "./render/render_{}.jpg".format(count)])
                    img = Image.open("./render/render_{}.jpg".format(count))
                    img_in = np.asarray(img)
                    batch_in = []
                    batch_in.append(img_in)
                    batch_out = []
                    batch_out.append(params)
                    feed_dict = {self.input: batch_in, self.target: batch_out}
                    loss, output = self.sess.run([self.loss, self.output], feed_dict=feed_dict)
                    total_loss += loss
                    count += 1
                except Exception as fail:
                    print fail
                    continue
            if fresh and epoch == 0:
                print "Image dataset rendered"
            print "Epoch: {} --> Average Loss: {}".format(epoch, total_loss / len(data))


def generate_samples(size):
    samples = []
    for count in range(size):
        sample = []
        #Generate shader parameters randomly
        sample.append(random.uniform(0.5, 1.5))             # Density Scale (0 - 2) default: 1
        sample.append(random.uniform(0.5, 1.5))             # Smoke Brightness (0 - 2) default: 1
        smoke_color = random.random()                   # Smoke Color (0 - 1) default: 0.2
        sample.append(smoke_color)
        sample.append(smoke_color)
        sample.append(smoke_color)
        sample.append(random.uniform(0.5, 1.5))             # Intensity Scale (0 - 5) default: 2
        sample.append(random.uniform(0, .4))           # Temperature Scale (0 - 5) default: 0.2
        sample.append(int(random.uniform(2500, 7500)))  # Color Temp in Kelvin (0 - 15000) default: 5000
        sample.append(random.uniform(0.05, 0.25))       # Adaption (0 - 1) default: 0.15
        sample.append(random.uniform(-0.5, 0.5))        # Burn (-2 - 2) default: 0
        samples.append(sample)
    return samples


def create_training_file(size=100):
    with open('./train_data/shader_params.csv', "w+") as f:
        writer = csv.writer(f)
        writer.writerows(generate_samples(size))
        f.close()


def main():
    param_learner = ParamLearner()
    param_learner.start_train(fresh=False)


if __name__ == '__main__':
    main()
