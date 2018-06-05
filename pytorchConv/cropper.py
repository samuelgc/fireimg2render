from PIL import Image
import numpy as np
import pprint as pp
def getSumAt(img,row,col,size):
    sumTot = 0
        # sumTot += np.sum(img[x-size//2:x+size//2,y-(size//2)+ySize,:])
    rowH,colH,depth = img.shape
    for rowSize in range(size):
        startCol = col-(size//2)
        endCol = col+((size+1)//2)
        if(row - (size//2) + rowSize < 0):
            continue
        if(row - (size//2) + rowSize > len(img) - 1):
            continue
        if(col-(size//2) < 0):
            startCol = 0
        # if(col+(size+1)//2 > colH):
        #     continue
        # print img[row - (size//2) + rowSize,startCol:endCol,:]
        sumTot += np.sum(img[row - (size//2) + rowSize,startCol:endCol,:])
    return sumTot

def testCrop():
    img = Image.open("./testFire.JPEG")
    # img.show()
    arr = np.asarray(img)
    x,y,z = arr.shape
    arrFill = np.empty([x,y])
    for xTe in range(x):
        for yTe in range(y):
            arrFill[xTe][yTe] = getSumAt(arr,xTe,yTe,32)
    print arrFill

    num = np.max(arrFill)
    numI = np.argmax(arrFill)
    realRow = numI // y
    realCol = numI % y
    print realRow, realCol
    img2 = img.crop((realCol - 16, realRow - 16,realCol + 16,realRow + 16))
    img2.show()
def cropAll():
    pass

if __name__== '__main__':
    testCrop()
    # stringManipulationTest()