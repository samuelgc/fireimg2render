import simplejson
import numpy as np
from PIL import Image
dimX = 35
dimY = 65
dimZ = 35
def loadJson(path):
    return simplejson.load(open(path,'r'))
def getInfo(data):
    geoData = data#loadJson("./help.json")

    dataInfo = geoData[11]
    dataType = dataInfo['volume_summary']
    working = dataType.splitlines()
    volumeSum = {}
    secLine = working[1]
    ind1 , ind2 = secLine.find("[") , secLine.find("]")
    arrDim = secLine[ind1+1: ind2].split(",")
    dimX , dimY , dimZ = arrDim[:]
    print arrDim , dimX , dimY , dimZ
    for i , line in enumerate(working):
        if(i == 0):
            continue
        ind1 , ind2 = line.find("(") , line.find(")")
        volumeSum[i - 1] = line[ind1 + 1:ind2]
    print volumeSum
    return volumeSum , int(dimX) , int(dimY) , int(dimZ)
def getTileData(tile,mat,xOff,yOff,zOff,comp):
    print dimX , dimY , dimZ
    print comp
    x_start = xOff * 16
    y_start = yOff * 16
    z_start = zOff * 16
    #lengths = 16
    xLen = 16
    yLen = 16
    zLen = 16
    #if not full tile make lengths real lengths
    if(comp == 0):
        xLen = dimX - xOff * 16
        yLen = dimY - yOff * 16
        zLen = dimZ - zOff * 16
    if xLen > 16:
        xLen = 16
    if yLen > 16:
        yLen = 16
    if zLen > 16:
        zLen = 16
    #make np array of data
    arr = np.array(tile)
    #make default ends
    x_end = x_start + xLen
    y_end = y_start + yLen
    z_end = z_start + zLen
    #if values are all the same
    if(comp == 2):
        print "entered 2"
        if x_end > dimX:
            print "entered X"
            x_end = dimX
            xLen = x_end - x_start 
        if y_end > dimY:
            print "entered Y"
            y_end = dimY
            yLen = y_end - y_start 
        if z_end > dimZ:
            print "entered Z"
            z_end = dimZ
            zLen = z_end - z_start 
        arr = np.full((xLen,yLen,zLen),0)        
    print "xs[{}] ys[{}] zs[{}]\nxe[{}] ye[{}] ze[{}]\nxl[{}] yl[{}] zl[{}]".format(x_start,y_start,z_start,x_end,y_end,z_end,xLen,yLen,zLen)

    arrMat = arr.reshape((xLen,yLen,zLen))
    # print mat[x_start:x_end,y_start:y_end,z_start:z_end].shape , arrMat.shape
    mat[x_start:x_end,y_start:y_end,z_start:z_end] = arrMat
def getVoxelDataAt(data,x,y,z):
    comprArr = data[2][1][3]
    tiles = data[2][1][5]
    mat = np.empty((x,y,z))
    # print len(tiles)
    cx , cy , cz = 0,0,0
    print "there are [{}] tiles".format(len(tiles))
    for i,tile in enumerate(tiles):
        print "tile number[{}]".format(i)
        getTileData(tile[3],mat,cx,cy,cz,tile[1])
        cx += 1
        if(cx > dimX // 16):
            cy += 1
            cx = 0
        if(cy > dimY // 16):
            cz += 1
            cy = 0
        if(cz > dimZ // 16):
            cz = 0
    return mat


def getAllVoxelData(geoData,geoInfo):
    allVoxelData = geoData[21]
    voxelData = {}
    for i in range(len(geoInfo)):
        voxelData[geoInfo[i]] = getVoxelDataAt(allVoxelData[i*2 + 1],dimX, dimY, dimZ)
    return voxelData

if __name__== '__main__':
    geoData = loadJson("./help.json")
    geoInfo, dimX , dimY , dimZ = getInfo(geoData)
    # for key in geoInfo.keys():
    #     print "key[{}]".format(geoInfo[key])
    # print dimX , dimY , dimZ
    voxelData = getAllVoxelData(geoData,geoInfo)
    for key in voxelData.keys():
        print "key[{}] unique[{}]".format(key,len(np.unique(voxelData[key])))
    density = voxelData["fuel"]
    slicedDen = density[:,:,17] * 255
    print np.unique(slicedDen)
    print slicedDen.shape
    img = Image.fromarray(slicedDen)
    img.show()


    
    # print geoInfo