import numpy as np
import random
import csv


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


def normalize_training_file():
    data = np.loadtxt('./train_data/shader_params.csv', delimiter=",")
    for count in range(len(data)):
        data[count, 0] /= 2.0
        data[count, 1] /= 2.0
        data[count, 5] /= 5.0
        data[count, 6] /= 5.0
        data[count, 7] /= 15000.0
        data[count, 9] /= 2.0
    with open('./train_data/normalized/shader_params.csv', "w+") as f:
        writer = csv.writer(f)
        writer.writerows(data)
        f.close()

