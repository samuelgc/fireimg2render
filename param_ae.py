import tensorflow as tf

from PIL import Image
from data_gen import *
from ifd_parse import *


def lrelu(x, alpha=0.1):
    return tf.maximum(alpha * x, x)


def clip(x):
    return tf.maximum(x, 0)


class ParamAutoEncoder:
    input_size = [None, 128, 128, 3]
    intrinsic_size = [None, 12]
    param_size = [None, 10]

    def __init__(self):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.input = tf.placeholder(tf.float32, self.input_size, name='input')
        self.intrinsic = tf.placeholder(tf.float32, self.intrinsic_size, name='intrinsic')
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
        flat_plus = tf.concat([flat, self.intrinsic], 1)
        dense0 = tf.layers.dense(flat_plus, units=1024, activation=lrelu, name='dense0')
        dense1 = tf.layers.dense(dense0, units=512, activation=lrelu, name='dense1')
        self.encoded = tf.layers.dense(dense1, units=10, name='encoded')

        #Decoding
        decode2 = tf.layers.dense(self.encoded, units=512, activation=lrelu, name='decode2')
        decode1 = tf.layers.dense(decode2, units=1024, activation=lrelu, name='decode1')
        decode0 = tf.layers.dense(decode1, units=16*16*32, activation=lrelu, name='decode0')

        #Deconvolution layers
        blowout = tf.reshape(decode0, [-1, 16, 16, 32])
        deconv2 = tf.layers.conv2d(blowout, 32, 3, padding="same", activation=lrelu, name='deconv2')
        upsamp2 = tf.layers.conv2d_transpose(deconv2, 32, 3, 2, padding="same", name='upsamp2')
        upsamp1 = tf.layers.conv2d_transpose(upsamp2, 48, 3, 2, padding="same", name='upsamp1')
        upsamp0 = tf.layers.conv2d_transpose(upsamp1, 64, 3, 2, padding="same", name='upsamp0')
        self.decoded = tf.layers.conv2d(upsamp0, 3, 3, activation=clip, padding="same", name='decoded')

        #Parameter Loss
        self.param_diff = tf.square(self.target - self.encoded)
        self.param_loss = tf.reduce_mean(self.param_diff, name='param_loss')
        self.param_train = tf.train.AdamOptimizer().minimize(self.param_loss, name='param_train')

        #Image Loss
        self.image_diff = tf.square(self.input - self.decoded)
        self.image_loss = tf.reduce_mean(self.image_diff, name='image_loss')
        self.image_train = tf.train.AdamOptimizer().minimize(self.image_loss, name='image_train')
        self.train = tf.group(self.param_train, self.image_train)

        #Summaries
        self.initial = tf.global_variables_initializer()
        tf.summary.image("input", self.input)
        tf.summary.image("result", self.decoded)
        tf.summary.tensor_summary("target", self.target)
        tf.summary.tensor_summary("output", self.encoded)
        tf.summary.scalar("param loss", self.param_loss)
        tf.summary.scalar("image loss", self.image_loss)
        self.merge = tf.summary.merge_all()

    def start_train(self, fresh=False, norm=True, sample_size=400, batch_size=10):
        if fresh:
            generate_data(sample_size)
            print "New training data generated"
        if norm:
            data = np.loadtxt('./train_data/normalized/shader_params.csv', delimiter=",")
        else:
            data = np.loadtxt('./train_data/shader_params.csv', delimiter=",")

        self.sess.run(self.initial)
        summary_write = tf.summary.FileWriter('/tmp/logs/ae_log', graph=tf.get_default_graph())

        x = 0
        change_count = 0
        last_mse = 0
        epoch = 0
        while change_count < 7:
            sample_set = np.arange(len(data))
            np.random.shuffle(sample_set)
            total_loss = 0
            batch_count = 0
            batch_in = []
            intrin_in = []
            batch_out = []
            for count in range(len(data)):
                if batch_count >= batch_size:
                    try:
                        feed_dict = {self.input: batch_in, self.intrinsic: intrin_in, self.target: batch_out}
                        summary, loss, _ = self.sess.run([self.merge, self.param_loss, self.train], feed_dict=feed_dict)
                        summary_write.add_summary(summary, x)
                        x += 1
                        total_loss += loss
                        batch_count = 0
                        batch_in = []
                        intrin_in = []
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
                    intrin_in.append(getIntrinsics("./ifds/fire_{}.ifd".format(int(item/400))))
                    batch_count += 1
            if len(batch_in) > 0:
                try:
                    feed_dict = {self.input: batch_in, self.intrinsic: intrin_in, self.target: batch_out}
                    loss, _ = self.sess.run([self.param_loss, self.train], feed_dict=feed_dict)
                    total_loss += loss
                except Exception as fail:
                    continue

            if fresh and epoch == 0:
                print "Image dataset rendered"

            epoch += 1
            mse = total_loss / len(data)
            if abs(mse - last_mse) < 0.0005:
                change_count += 1
            else:
                change_count = 0
            last_mse = mse
            print "Epoch: {} --> Average Loss: {}".format(epoch, total_loss / len(data))


def main():
    auto_encoder = ParamAutoEncoder()
    auto_encoder.start_train()


if __name__ == '__main__':
    main()