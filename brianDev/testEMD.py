from scipy.stats import wasserstein_distance
import os
import random
from PIL import Image
import numpy as np

def showScores():
    # fileDir = "../fireTemp/"
    fileDir = "../cropped/mask/"
    files = os.listdir(fileDir)
    files2 = os.listdir(fileDir)
    random.shuffle(files2)
    for i,(filename1,filename2) in enumerate(zip(files,files2)):
        img = Image.open(fileDir + filename1)
        img2 = Image.open(fileDir + filename2)
        h1 = img.histogram()
        h2 = img2.histogram()
        # emdScore = pyemd.emd_samples(img,img2)
        wasserstein_distance_score = wasserstein_distance(h1,h2)
        print "CHECK[{}] EMD_SCORE[{}]".format(i,wasserstein_distance_score)
        img.show()
        img2.show()
        inputYo = True
        while(inputYo):
            inputYo = input()
            
        img.close()
        img2.close()
        # if i == 0:
            # break
def changeColor(img,x,y ,size,colorArr):
    for row in range(size):
        for col in range(size):
            img[row+x][col+y] = colorArr
def createStupidImg():
    # for i in range(50):
    npArr = np.array([0]*30000)
    lol = np.reshape(npArr,(100,100,3))
    changeColor(lol,0,0,20,[255,0,0])

    npArr2 = np.array([0]*30000)
    lol2 = np.reshape(npArr2,(100,100,3))
    changeColor(lol2,50,50,20,[255,0,0])

    img = Image.fromarray(np.uint8(lol))
    img2 = Image.fromarray(np.uint8(lol2))
    h1 = img.histogram()
    h2 = img2.histogram()
    # print h1
    # print h2
    wasserstein_distance_score = wasserstein_distance(h1,h2)

    #emdScore = pyemd.emd_samples(np.reshape(lol,(30000)),np.reshape(lol2,(30000)))
    img.show()
    img2.show()
    print wasserstein_distance_score
if __name__== '__main__':
    # showScores()
    createStupidImg()
    # stringManipulationTest()