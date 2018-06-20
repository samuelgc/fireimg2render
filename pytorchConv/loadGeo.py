import simplejson
import numpy as np
from PIL import Image
import collections
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
    for i , line in enumerate(working):
        if(i == 0):
            continue
        ind1 , ind2 = line.find("(") , line.find(")")
        volumeSum[i - 1] = line[ind1 + 1:ind2]
    print volumeSum
    return volumeSum , int(dimX) , int(dimY) , int(dimZ)
def printBy16(tile):
    for st in range(len(tile)//16 - 1):
        print st , tile[st*16:(st+1)*16]
def getTileData(tile,mat,xOff,yOff,zOff,comp):
    # print comp
    # Calculating start x y z values for the tile
    x_start = xOff * 16
    y_start = yOff * 16
    z_start = zOff * 16
    # Calculate the lengths if on edge AKA where lengths < 16
    xLen = dimX - xOff * 16
    yLen = dimY - yOff * 16
    zLen = dimZ - zOff * 16
    # if not on edge reset to 16
    if xLen > 16:
        xLen = 16
    if yLen > 16:
        yLen = 16
    if zLen > 16:
        zLen = 16
    #make np array of data
    arr = np.array(tile)
    arr2 = np.array(tile)
    #make ends based on previus calculated length and starts
    x_end = x_start + xLen
    y_end = y_start + yLen
    z_end = z_start + zLen
    #if values are all the same AKA compressed 
    if(comp == 2):
        arr = np.full((xLen,yLen,zLen),tile) # Both of these methods work 
        arr2 = np.full(xLen*yLen*zLen,tile) # Both of these methods work 
    #method one
    arrMat = arr.reshape((xLen,yLen,zLen))
    #method two
    arrLol = np.empty((xLen,yLen,zLen))
    count = 0
    for x in range(xLen):
        for y in range(yLen):
            for z in range(zLen):
                arrLol[x,y,z] = arr2[count]
                count += 1
    
    mat[x_start:x_end,y_start:y_end,z_start:z_end] = arrLol 
    # mat[x_start:x_end,y_start:y_end,z_start:z_end] = arrMat
def getVoxelDataAt(data,x,y,z):
    comprArr = data[2][1][3]
    tiles = data[2][1][5]
    mat = np.empty((x,y,z))
    tileCount = 0
    for tx in range((dimX + 15) // 16):
        for ty in range((dimY + 15) // 16):
            for tz in range((dimZ + 15) // 16):
                tile = tiles[tileCount]
                getTileData(tile[3],mat,tx,ty,tz,tile[1])
                tileCount +=1
    return mat


def getAllVoxelData(geoData,geoInfo):
    allVoxelData = geoData[21]
    voxelData = {}
    for i in range(len(geoInfo)):
        voxelData[geoInfo[i]] = getVoxelDataAt(allVoxelData[i*2 + 1],dimX, dimY, dimZ)
    return voxelData

if __name__== '__main__':
    geoData = loadJson("./temp.json")
    geoInfo, dimX , dimY , dimZ = getInfo(geoData)
    voxelData = getAllVoxelData(geoData,geoInfo)
    density = voxelData["density"]
    slicedDen = density[:,0,:]
    for sl in range(len(density)):
        imgTemp = Image.fromarray(np.uint8(density[:,sl,:] * 255),"L")
        imgTemp.save("slice[{}].png".format(str(sl + 1).zfill(2)))