import numpy as np
import random
import csv

from subprocess import call


def generate_samples(size):
    samples = []
    for count in range(size):
        sample = []
        #Generate shader parameters randomly
        sample.append(random.uniform(0.0, 2.0))             # Density Scale (0 - 2) default: 1
        sample.append(random.uniform(0.0, 2.0))             # Smoke Brightness (0 - 2) default: 1
        sample.append(random.random())                      # Smoke Color (0 - 1) default: 0.2
        sample.append(random.random())
        sample.append(random.random())
        sample.append(random.uniform(0.0, 5.0))             # Intensity Scale (0 - 5) default: 2
        sample.append(random.uniform(0.0, 5.0))             # Temperature Scale (0 - 5) default: 0.2
        sample.append(int(random.uniform(0, 7500)))        # Color Temp in Kelvin (0 - 15000) default: 5000
        sample.append(random.uniform(0.0, 1.0))             # Adaption (0 - 1) default: 0.15
        sample.append(random.uniform(-2.0, 2.0))            # Burn (-2 - 2) default: 0
        samples.append(sample)
    return samples


def create_training_file(size=100):
    with open('./random/training.csv', "w+") as f:
        writer = csv.writer(f)
        writer.writerows(generate_samples(size))
        f.close()


def add_to_training_file(selection, size=100):
    with open('./train_data/shader_params.csv', "a") as f:
        writer = csv.writer(f)
        data = generate_samples(size)
        writer.writerows(data)
        f.close()
        print "{} samples generated".format(size)
        for i in range(size):
            print "Rendering image {} of {}".format(i + 1, size)
            render_sample(selection, i, data[i])
        data = normalize_data(data)
        with open('./train_data/normalized/shader_params.csv', "a") as f:
            writer = csv.writer(f)
            writer.writerows(data)
            f.close()
        print "Added set normalized"


def normalize_data(data):
    for count in range(len(data)):
        params = data[count]
        params[0] /= 2.0
        params[1] /= 2.0
        params[5] /= 5.0
        params[6] /= 5.0
        params[7] /= 7500
        params[9] += 2.0
        params[9] /= 4.0
    return data


def normalize_training_file():
    data = np.loadtxt('./random/training.csv', delimiter=",")
    data = normalize_data(data)
    with open('./random/n_training.csv', "w+") as f:
        writer = csv.writer(f)
        writer.writerows(data)
        f.close()


def denormalize(sample):
    sample[0] *= 2.0
    sample[1] *= 2.0
    sample[5] *= 5.0
    sample[6] *= 5.0
    sample[7] *= 7500
    sample[9] *= 4.0
    sample[9] -= 2.0
    for x in range(len(sample) - 1):
        if sample[x] < 0:
            sample[x] = 0
    return sample


def render_sample(select, item, params):
    with open('./ifds/fire_{}.ifd'.format(select)) as f:
        search_string = "fc_colorramp_the_basis_strings ( \"linear\" \"linear\" ) fc_colorramp_the_key_positions ( 0 1 ) fc_colorramp_the_key_values ( 0 0 0 1 1 1 )"
        replace_string = "s_densityscale {} s_int {} s_color {} {} {} fi_int {} fc_int {} fc_colorramp_the_basis_strings ( \"linear\" \"linear\" ) fc_colorramp_the_key_positions ( 0 1 ) fc_colorramp_the_key_values ( 0 0 0 1 1 1 ) fc_bbtemp {} fc_bbadapt {} fc_bbburn {}" \
            .format(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8],
                    params[9])
        contents = f.read().replace(search_string, replace_string)
    with open('./random/ifds/render_fire_{}.ifd'.format(item), "w+") as f:
        f.write(contents)
    call(["mantra", "./random/ifds/render_fire_{}.ifd".format(item), "./random/imgs/render_{}.jpg".format(item)])


def render_all_samples():
    data = np.loadtxt('./random/training.csv', delimiter=",")
    for i in range(len(data)):
        print "Rendering image {} of {}".format(i+1, len(data))
        render_sample(random.randint(0, 3), i, data[i])


def generate_data(size):
    create_training_file(size)
    print "New training set created"
    normalize_training_file()
    print "Training set normalized"
    render_all_samples()
    print "Training images rendered"


def main():
    generate_data(1000)


if __name__ == '__main__':
    main()