from subprocess import call
from PIL import Image
from data_gen import *

import tensorflow as tf


def lrelu(x, alpha=0.1):
    return tf.maximum(alpha * x, x)


class ParamLearner:
    input_size = [None, 128, 128, 3]
    output_size = [None, 10]

    def __init__(self):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.input = tf.placeholder(tf.float32, self.input_size, name='input')
        self.target = tf.placeholder(tf.float32, self.output_size, name='target')

        #Convolutional layers
        conv0 = tf.layers.conv2d(self.input, 64, 5, padding="same", activation=lrelu, name='conv0')
        pool0 = tf.layers.max_pooling2d(conv0, 2, 2, padding="same", name='pool0')
        conv1 = tf.layers.conv2d(pool0, 48, 5, padding="same", activation=lrelu, name='conv1')
        pool1 = tf.layers.max_pooling2d(conv1, 2, 2, padding="same", name='pool1')
        conv2 = tf.layers.conv2d(pool1, 32, 5, padding="same", activation=lrelu, name='conv2')
        pool2 = tf.layers.max_pooling2d(conv2, 2, 2, padding="same", name='pool2')

        #Dense layers
        flat = tf.layers.flatten(pool2)
        dense0 = tf.layers.dense(flat, units=2048, activation=lrelu, name='dense0')
        dense1 = tf.layers.dense(dense0, units=1024, activation=lrelu, name='dense1')
        dense2 = tf.layers.dense(dense1, units=512, activation=lrelu, name='dense2')
        self.output = tf.layers.dense(dense2, units=10, name='output')

        #Loss
        self.diff = tf.squared_difference(self.target, self.output)
        self.loss = tf.reduce_sum(self.diff, name='loss')
        self.cost = tf.reduce_mean(self.diff)
        self.train = tf.train.AdamOptimizer().minimize(self.cost, name='train')

        #Summaries
        self.initial = tf.global_variables_initializer()
        tf.summary.image("input", self.input)
        tf.summary.tensor_summary("target", self.target)
        tf.summary.tensor_summary("output", self.output)
        tf.summary.scalar("loss", self.cost)
        self.merge = tf.summary.merge_all()

    def start_train(self, fresh=False, norm=True, sample_size=500, batch_size=10):
        if fresh:
            generate_data(sample_size)
            print "New training data generated"
        if norm:
            data = np.loadtxt('./train_data/normalized/shader_params.csv', delimiter=",")
        else:
            data = np.loadtxt('./train_data/shader_params.csv', delimiter=",")

        self.sess.run(self.initial)
        summary_write = tf.summary.FileWriter('/tmp/logs/one_log', graph=tf.get_default_graph())

        change_count = 0
        last_mse = 0
        epoch = 0
        while change_count < 7:
            sample_set = np.arange(len(data))
            if (fresh and epoch > 0) or not fresh:
                np.random.shuffle(sample_set)
            total_loss = 0
            batch_count = 0
            batch_in = []
            batch_out = []
            for count in range(len(data)):
                if batch_count >= batch_size:
                    try:
                        feed_dict = {self.input: batch_in, self.target: batch_out}
                        summary, loss, output, _ = self.sess.run([self.merge, self.cost, self.output, self.train], feed_dict=feed_dict)
                        summary_write.add_summary(summary, epoch)
                        total_loss += loss
                        batch_count = 0
                        batch_in = []
                        batch_out = []
                    except Exception as fail:
                        continue
                else:
                    item = sample_set[count]
                    params = data[item]
                    img = Image.open("./render/render_{}.jpg".format(item))
                    img.thumbnail((128, 128), Image.ANTIALIAS)
                    img_in = np.asarray(img)
                    img_in = img_in / 255.0
                    count += 1
                    batch_in.append(img_in)
                    batch_out.append(params)
                    batch_count += 1
            if len(batch_in) > 0:
                try:
                    feed_dict = {self.input: batch_in, self.target: batch_out}
                    loss, output, _ = self.sess.run([self.cost, self.output, self.train], feed_dict=feed_dict)
                    total_loss += loss
                except Exception as fail:
                    continue

            if fresh and epoch == 0:
                print "Image dataset rendered"

            epoch += 1
            mse = total_loss / len(data)
            if abs(mse - last_mse) < 0.0002:
                change_count += 1
            else:
                change_count = 0
            last_mse = mse
            print "Epoch: {} --> Average Loss: {}".format(epoch, mse)

    def test(self, filename, attempt):
        img = Image.open(filename)
        img = img.resize((128, 128), Image.BICUBIC)
        # img.thumbnail((128, 128), Image.ANTIALIAS)
        img_in = np.asarray(img)
        img_in = img_in / 255.0
        batch_in = []
        batch_in.append(img_in)
        feed_dict = {self.input: batch_in}
        output, _ = self.sess.run([self.output, self.input], feed_dict=feed_dict)

        params = denormalize(output[0])
        with open('./ifds/fire.ifd') as f:
            search_string = "fc_colorramp_the_basis_strings ( \"linear\" \"linear\" ) fc_colorramp_the_key_positions ( 0 1 ) fc_colorramp_the_key_values ( 0 0 0 1 1 1 )"
            replace_string = "s_densityscale {} s_int {} s_color {} {} {} fi_int {} fc_int {} fc_colorramp_the_basis_strings ( \"linear\" \"linear\" ) fc_colorramp_the_key_positions ( 0 1 ) fc_colorramp_the_key_values ( 0 0 0 1 1 1 ) fc_bbtemp {} fc_bbadapt {} fc_bbburn {}" \
                .format(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9])
            contents = f.read().replace(search_string, replace_string)
        with open('./ifds/test_render{}.ifd'.format(attempt), "w+") as f:
            f.write(contents)
        call(["mantra", "./ifds/test_render{}.ifd".format(attempt), "./render/test_render{}.jpg".format(attempt)])


def main():
    param_learner = ParamLearner()
    param_learner.start_train()
    param_learner.test("./fire_images/google/fire/19. fire_from_brazier.jpg", 0)
    param_learner.test("./render/render_56.jpg", 1)


if __name__ == '__main__':
    main()
