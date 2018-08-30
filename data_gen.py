import numpy as np
import random
import csv

from subprocess import call


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
        sample.append(random.uniform(0.0, 0.4))           # Temperature Scale (0 - 5) default: 0.2
        sample.append(int(random.uniform(4000, 6000)))  # Color Temp in Kelvin (0 - 15000) default: 5000
        sample.append(random.uniform(0.05, 0.25))       # Adaption (0 - 1) default: 0.15
        sample.append(random.uniform(-0.5, 0.5))        # Burn (-2 - 2) default: 0
        samples.append(sample)
    return samples


def create_training_file(size=100):
    with open('./train_data/shader_params.csv', "w+") as f:
        writer = csv.writer(f)
        writer.writerows(generate_samples(size))
        f.close()


def add_to_training_file(size=100):
    with open('./train_data/shader_params.csv', "a") as f:
        writer = csv.writer(f)
        data = generate_samples(size)
        writer.writerows(data)
        f.close()
        print "{} samples generated".format(size)
        for i in range(size):
            print "Rendering image {} of {}".format(i + 1, size)
            render_sample(0, 0, i, data[i])
        normalize_data(data)
        print "Added set normalized"


def normalize_data(data):
    for count in range(len(data)):
        data[count, 0] -= 0.5
        data[count, 1] -= 0.5
        data[count, 5] -= 0.5
        data[count, 6] /= 0.4
        data[count, 7] -= 2500
        data[count, 7] /= 5000
        data[count, 8] -= 0.05
        data[count, 8] /= 0.2
        data[count, 9] += 0.5
    with open('./train_data/normalized/shader_params.csv', "a") as f:
        writer = csv.writer(f)
        writer.writerows(data)
        f.close()


def normalize_training_file():
    data = np.loadtxt('./train_data/shader_params.csv', delimiter=",")
    for count in range(len(data)):
        data[count, 0] -= 0.5
        data[count, 1] -= 0.5
        data[count, 5] -= 0.5
        data[count, 6] /= 0.4
        data[count, 7] -= 2500
        data[count, 7] /= 5000
        data[count, 8] -= 0.05
        data[count, 8] /= 0.2
        data[count, 9] += 0.5
    with open('./train_data/normalized/shader_params.csv', "w+") as f:
        writer = csv.writer(f)
        writer.writerows(data)
        f.close()


def denormalize(sample):
    sample[0] += 0.5
    sample[1] += 0.5
    sample[5] += 0.5
    sample[6] *= 0.4
    sample[7] *= 5000
    sample[7] += 2500
    sample[8] *= 0.2
    sample[8] += 0.05
    sample[9] -= 0.5
    return sample


def render_sample(select, epoch, item, params):
    with open('./ifds/fire_{}.ifd'.format(select)) as f:
        search_string = "fc_colorramp_the_basis_strings ( \"linear\" \"linear\" ) fc_colorramp_the_key_positions ( 0 1 ) fc_colorramp_the_key_values ( 0 0 0 1 1 1 )"
        replace_string = "s_densityscale {} s_int {} s_color {} {} {} fi_int {} fc_int {} fc_colorramp_the_basis_strings ( \"linear\" \"linear\" ) fc_colorramp_the_key_positions ( 0 1 ) fc_colorramp_the_key_values ( 0 0 0 1 1 1 ) fc_bbtemp {} fc_bbadapt {} fc_bbburn {}" \
            .format(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8],
                    params[9])
        contents = f.read().replace(search_string, replace_string)
    with open('./ifds/render_fire_{}_{}.ifd'.format(epoch, item), "w+") as f:
        f.write(contents)
    call(["mantra", "./ifds/render_fire_{}_{}.ifd".format(epoch, item), "./render/render_{}.jpg".format(item)])


def render_all_samples():
    data = np.loadtxt('./train_data/shader_params.csv', delimiter=",")
    for i in range(len(data)):
        print "Rendering image {} of {}".format(i+1, len(data))
        render_sample(random.randint(0, 3), 0, i, data[i])


def generate_data(size):
    create_training_file(size)
    print "New training set created"
    normalize_training_file()
    print "Training set normalized"
    render_all_samples()
    print "Training images rendered"


def main():
    add_to_training_file()


if __name__ == '__main__':
    main()