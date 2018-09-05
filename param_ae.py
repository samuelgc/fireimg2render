import tensorflow as tf
from arg_help import *
from PIL import Image
from data_gen import *
import sys


def lrelu(x, alpha=0.1):
    return tf.maximum(alpha * x, x)
def clip(x):
    return tf.maximum(x,0)

class ParamAutoEncoder:
    input_size = [None, 128, 128, 3]
    param_size = [None, 10]

    def __init__(self,args):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.input = tf.placeholder(tf.float32, self.input_size, name='input')
        self.target = tf.placeholder(tf.float32, self.param_size, name='target')
        l1 = args.l1
        l2 = args.l2
        l3 = args.l3
        k1,k2,k3 = args.k1,args.k2,args.k3
        #Convolution layers
        conv0 = tf.layers.conv2d(self.input, l1, k1, padding="same", activation=lrelu, name='conv0')
        pool0 = tf.layers.max_pooling2d(conv0, 2, 2, padding="same", name='pool0')
        conv1 = tf.layers.conv2d(pool0, l2, k2, padding="same", activation=lrelu, name='conv1')
        pool1 = tf.layers.max_pooling2d(conv1, 2, 2, padding="same", name='pool1')
        conv2 = tf.layers.conv2d(pool1, l3, k3, padding="same", activation=lrelu, name='conv2')
        pool2 = tf.layers.max_pooling2d(conv2, 2, 2, padding="same", name='pool2')

        #Dense layers
        flat = tf.reshape(pool2, [-1, 16*16*l3])
        dense0 = tf.layers.dense(flat, units=1024, activation=lrelu, name='dense0')
        dense1 = tf.layers.dense(dense0, units=512, activation=lrelu, name='dense1')
        self.encoded = tf.layers.dense(dense1, units=10, name='encoded')

        #Decoding

        decode1 = tf.layers.dense(self.encoded, units=1024, activation=lrelu, name='decode1')
        decode0 = tf.layers.dense(decode1, units=16*16*l3, activation=lrelu, name='decode0')

        #Deconvolution layers
        blowout = tf.reshape(decode0, [-1, 16, 16, l3])
        deconv2 = tf.layers.conv2d(blowout, 32, 3, padding="same", activation=lrelu, name='deconv2')
        upsamp2 = tf.layers.conv2d_transpose(deconv2, l3, 3, 2, padding="same", name='upsamp2')
        upsamp1 = tf.layers.conv2d_transpose(upsamp2, l2, 3, 2, padding="same", name='upsamp1')
        upsamp0 = tf.layers.conv2d_transpose(upsamp1, l1, 3, 2, padding="same", name='upsamp0')
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
        #debuggin
        self.save_img = self.decoded
        #Summaries
        self.initial = tf.global_variables_initializer()
        tf.summary.image("input", self.input)
        tf.summary.image("result", self.decoded)
        tf.summary.scalar("min", tf.reduce_min(self.decoded))
        tf.summary.scalar("max", tf.reduce_max(self.decoded))
        tf.summary.tensor_summary("target", self.target)
        tf.summary.tensor_summary("output", self.encoded)
        tf.summary.scalar("loss/param_loss", self.param_loss)
        tf.summary.scalar("loss/image_loss", self.image_loss)
        self.merge = tf.summary.merge_all()

    def start_train(self, fresh=False, norm=True, sample_size=100, batch_size=10, epochs=100,expname="default"):
        if fresh:
            generate_data(sample_size)
            print "New training data generated"
        if norm:
            data = np.loadtxt('./train_data/normalized/shader_params.csv', delimiter=",")
            print "Using Normalized saved files"
        else:
            data = np.loadtxt('./train_data/shader_params.csv', delimiter=",")
            print "Using Non-Normalized saved files"

        self.sess.run(self.initial)
        summary_write = tf.summary.FileWriter('tb/{}'.format(expname), graph=tf.get_default_graph())

        for epoch in range(epochs):
            sample_set = np.arange(sample_size)
            np.random.shuffle(sample_set)
            total_loss = 0
            batch_count = 0
            batch_in = []
            batch_out = []
            badcount = 0
            for count in range(len(data)):
                if batch_count >= batch_size:
                    try:
                        feed_dict = {self.input: batch_in, self.target: batch_out}
                        summary,img_save, loss, _ = self.sess.run([self.merge,self.save_img,self.param_loss, self.train], feed_dict=feed_dict)
                        summary_write.add_summary(summary, epoch)
                        total_loss += loss
                        batch_count = 0
                        batch_in = []
                        batch_out = []
                    except Exception as fail:
                        print str(fail)
                        continue
                else:
                    item = sample_set[count]
                    params = data[item]
                    img = Image.open("./render/render_{}.jpg".format(item))
                    # img.thumbnail((128, 128), Image.ANTIALIAS)
                    img = img.resize((128,128),Image.BICUBIC)
                    img_in = np.asarray(img)
                    img_in = img_in / 255.0
                    count += 1
                    batch_in.append(img_in)
                    batch_out.append(params)
                    batch_count += 1
            if len(batch_in) > 0:
                try:
                    feed_dict = {self.input: batch_in, self.target: batch_out}
                    loss, _ = self.sess.run([self.param_loss, self.train], feed_dict=feed_dict)
                    total_loss += loss
                except Exception as fail:
                    print str(fail)
                    continue

            if fresh and epoch == 0:
                print "Image dataset rendered"
            print "Epoch: {} --> Average Loss: {}".format(epoch, total_loss / len(data))


def main():
    args = load_args(sys.argv[1])
    # for arg in args:
        # print arg," = ", args[arg]
    auto_encoder = ParamAutoEncoder(args)
    auto_encoder.start_train(expname=args.expname)


if __name__ == '__main__':
    main()