from subprocess import call
import sys
from mask_fire import *
import math
import cPickle as pickle
# import os

numero = 0 

def l_relu(x):
    result = []
    for i in range(len(x)):
        if x[i] <= 0:
            result.append(0.01 * x[i])
        else:
            result.append(x[i] * (x[i] > 0))
    return result


def d_lrelu(x):
    result = []
    for i in range(len(x)):
        if x[i] <= 0:
            result.append(0.01)
        else:
            result.append(1.0)
    return result

def sigmoid(x):
    result = []
    for num in x:
        result.append(1 / (1 + math.exp(-num)))
    return result
def d_sigmoid(x):
    # print(x)
    result = x * (1-x)
    return result


class MLP:
    """
    Multi Layer Perceptron
    Usage: args is the shape of the model
    i.e. [ 3, 4, 4, 2 ] is an MLP with 3 inputs two hidden layers of 4 each and 2 outputs
    """
    def __init__(self, *args):
        self.shape = args
        self.numeroSave = 0
        n = len(args)

        # Build layers
        self.layers =[]
        self.layers.append(np.ones(self.shape[0]+1))
        for i in range(1,n):
            self.layers.append(np.ones(self.shape[i]))

        # Build weights
        self.weights = []
        for i in range(n-1):
            self.weights.append(np.zeros((self.layers[i].size,
                                          self.layers[i+1].size)))

        # Last change in weights for momentum
        self.dw = [0,]*len(self.weights)

        # Reset weights
        self.reset(0.1)

    def reset(self, wid):
        for i in range(len(self.weights)):
            Z = np.random.random((self.layers[i].size, self.layers[i+1].size))
            self.weights[i][...] = (2*Z-1)*wid

    def step_forward(self, data):
        self.layers[0][0:-1] = data

        # for i in range(1,len(self.shape)):
        #     self.layers[i][...] = sigmoid(np.dot(self.layers[i-1], self.weights[i-1]))
        # self.layers[-1] = np.clip(self.layers[-1],0,1)
        # attempt with all l_relu but last one
        shapelen = len(self.shape)
        for i in range(1,len(self.shape)-1):
            self.layers[i][...] = l_relu(np.dot(self.layers[i-1], self.weights[i-1]))
        self.layers[shapelen -1][...] = sigmoid(np.dot(self.layers[shapelen - 2], self.weights[shapelen-2]))
        return self.layers[-1]

    def backprop(self, result, rate=0.1, mom=0.1):
        deltas = []

        error = np.mean(self.layers[0] - result) * -1
        delta = error * np.array(d_sigmoid(self.layers[-1]))
        deltas.append(delta)

        # Compute error
        for i in range(len(self.shape)-2, 0, -1):
            delta = np.dot(deltas[0], self.weights[i].T) * d_lrelu(self.layers[i])
            deltas.insert(0, delta)

        # Update weights
        for i in range(len(self.weights)):
            layer = np.atleast_2d(self.layers[i])
            delta = np.atleast_2d(deltas[i])
            dw = np.dot(layer.T, delta)
            self.weights[i] += rate * dw + mom * self.dw[i]
            self.dw[i] = dw

        return error

    def train(self, data, epochs=1000, lrate=0.1, mome=0.1,saveEpochs = 5,filename = "save.p"):
        for e in range(epochs):
            n = np.random.randint(len(data))
            temp = self.step_forward(data[n])
            fire_stats , heat1 = map_render(temp)
            fire_stats.append(1)
            error = self.backprop(fire_stats, rate=lrate, mom=mome)
            print "Epoch {}: Temperature output: {}, with an error of {}".format(e, heat1, error)
            if(e % saveEpochs  == 0):
                self.numeroSave = numero
                pickle.dump(self, open(filename,"wb"))
                print "Epoch {}: Model Saved to {} : numero {}".format(e,filename,self.numeroSave)


    def predict(self, data):
        return self.step_forward(data)


def render(heat):
    # Render the image...
    global numero
    numero = numero + 1
    with open('./ifds/fire.ifd') as f:
        contents = f.read().replace('bbtemp 4177', 'bbtemp ' + str(heat)).replace('bbtemp = 5000', 'bbtemp = ' + str(heat))
    with open('./ifds/render_fire_{}.ifd'.format(numero), "w+") as f:
        f.write(contents)
    f = open('/dev/null', 'w') # Added this so the warning about licencse expiring doesn't show up
    call(["mantra", "./ifds/render_fire_{}.ifd".format(numero), "./render/render_{}.jpg".format(numero)],stderr=f)

def map_render(temperature):
    # print temperature
    heat = temperature[0]
    if heat < 0:
        heat = 0
    if heat > 1:
        heat = 1
    heat *= 25000
    render(heat)

    fire_img = Image.open("./render/render_{}.jpg".format(numero))
    fire_mask = mask(fire_img)
    fire_stats = map_stats(fire_img, fire_mask)
    for i in range(len(fire_stats)):
        fire_stats[i] /= 25000
    return fire_stats, heat


def main():
    #print sys.argv[0] , sys.argv[1]
    global numero
    if(len(sys.argv) > 2):
        learner = pickle.load(open(sys.argv[2],"rb"))
        print "Loaded Check Point"
        numero = learner.numeroSave
    else:
        print "No Check Point"
        learner = MLP(9, 20, 30, 20, 1)
    # Train
    data = np.loadtxt('./train_data/normalized/google_fire.csv', delimiter=",")
    if(len(sys.argv) > 1):
        learner.train(data, 1000, 0.1, 0,filename = sys.argv[1])
    else:
        learner.train(data, 1000, 0.1, 0)


if __name__ == '__main__':
    main()
