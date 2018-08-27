import tensorflow as tf

from subprocess import call
from PIL import Image
from data_gen import *


def lrelu(x, alpha=0.1):
    return tf.maximum(alpha * x, x)


class ParamAutoEncoder:
    input_size = [None, 128, 128, 3]
    param_size = [None, 10]

    def __init__(self):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.input = tf.placeholder(tf.float32, self.input_size, name='input')
        self.target = tf.placeholder(tf.float32, self.param_size, name='target')

        #Convolution layers
        conv0 = tf.layers.conv2d(self.input, 64, 3, padding="same", activation=lrelu, name='conv0')
        pool0 = tf.layers.max_pooling2d(conv0, 2, 2, padding="same", name='pool0')
        conv1 = tf.layers.conv2d(pool0, 48, 3, padding="same", activation=lrelu, name='conv1')
        pool1 = tf.layers.max_pooling2d(conv1, 2, 2, padding="same", name='pool1')
        conv2 = tf.layers.conv2d(pool1, 32, 3, padding="same", activation=lrelu, name='conv2')
        pool2 = tf.layers.max_pooling2d(conv2, 2, 2, padding="same", name='pool2')

        #Dense layers
        flat = tf.reshape(pool2, [-1, 16*16*32])
        dense0 = tf.layers.dense(flat, units=1024, activation=lrelu, name='dense0')
        dense1 = tf.layers.dense(dense0, units=512, activation=lrelu, name='dense1')
        self.encoded = tf.layers.dense(dense1, units=10, name='encoded')

        #Decoding
        decode1 = tf.layers.dense(self.encoded, units=1024, activation=lrelu, name='decode1')
        decode0 = tf.layers.dense(decode1, units=16*16*32, activation=lrelu, name='decode0')

        #Deconvolution layers
        blowout = tf.reshape(decode0, [-1, 16, 16, 32])
        deconv2 = tf.layers.conv2d(blowout, 32, 3, padding="same", activation=lrelu, name='deconv2')
        upsamp2 = tf.layers.conv2d_transpose(deconv2, 32, 3, 2, padding="same", name="upsamp2")
        upsamp1 = tf.layers.conv2d_transpose(upsamp2, 48, 3, 2, padding="same", name='upsamp1')
        upsamp0 = tf.layers.conv2d_transpose(upsamp1, 64, 3, 2, padding="same", name='upsamp0')
        self.decoded = tf.layers.conv2d(upsamp0, 3, 3, padding="same", name='decoded')

        #Loss
        self.diff = tf.square(self.input - self.decoded)
        self.loss = tf.reduce_sum(self.diff, name='loss')
        self.cost = tf.reduce_mean(self.diff)
        self.train = tf.train.AdamOptimizer().minimize(self.cost, name='train')

        #Summaries
        self.initial = tf.global_variables_initializer()
        tf.summary.image("input", self.input)
        tf.summary.image("result", self.decoded)
        tf.summary.scalar("min", tf.reduce_min(self.decoded))
        tf.summary.scalar("max", tf.reduce_max(self.decoded))
        tf.summary.tensor_summary("target", self.target)
        tf.summary.tensor_summary("output", self.encoded)
        tf.summary.scalar("loss", self.cost)
        self.merge = tf.summary.merge_all()

    def start_train(self, fresh=True, norm=False, sample_size=500, batch_size=10, epochs=1000):
        if fresh:
            create_training_file(sample_size)
            print "New training file generated"
            if norm:
                normalize_training_file()
        if norm:
            data = np.loadtxt('./train_data/normalized/shader_params.csv', delimiter=",")
        else:
            data = np.loadtxt('./train_data/shader_params.csv', delimiter=",")

        self.sess.run(self.initial)
        summary_write = tf.summary.FileWriter('/tmp/logs/ae_log', graph=tf.get_default_graph())

        for epoch in range(epochs):
            sample_set = np.arange(sample_size)
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
                        summary, loss, _ = self.sess.run([self.merge, self.loss, self.train], feed_dict=feed_dict)
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
                    if fresh and epoch == 0:
                        with open('./ifds/fire.ifd') as f:
                            search_string = "fc_colorramp_the_basis_strings ( \"linear\" \"linear\" ) fc_colorramp_the_key_positions ( 0 1 ) fc_colorramp_the_key_values ( 0 0 0 1 1 1 )"
                            replace_string = "s_densityscale {} s_int {} s_color {} {} {} fi_int {} fc_int {} fc_colorramp_the_basis_strings ( \"linear\" \"linear\" ) fc_colorramp_the_key_positions ( 0 1 ) fc_colorramp_the_key_values ( 0 0 0 1 1 1 ) fc_bbtemp {} fc_bbadapt {} fc_bbburn {}" \
                                .format(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9])
                            contents = f.read().replace(search_string, replace_string)
                        with open('./ifds/render_fire_{}_{}.ifd'.format(epoch, item), "w+") as f:
                            f.write(contents)
                        call(["mantra", "./ifds/render_fire_{}_{}.ifd".format(epoch, item), "./render/render_{}.jpg".format(item)])
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
                    loss, _ = self.sess.run([self.loss, self.train], feed_dict=feed_dict)
                    total_loss += loss
                except Exception as fail:
                    continue

            if fresh and epoch == 0:
                print "Image dataset rendered"
            print "Epoch: {} --> Average Loss: {}".format(epoch, total_loss / len(data))


def main():
    auto_encoder = ParamAutoEncoder()
    auto_encoder.start_train(fresh=False, norm=True)


if __name__ == '__main__':
    main()