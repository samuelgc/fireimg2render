import simplejson
import numpy as np


def pullGeo(filename):
    f = open(filename, "r")
    file_str = f.read()
    start_ind = file_str.find("ray_detail")
    end_ind = file_str.find("]\nray_end")
    geoInfo = file_str[start_ind:end_ind + 2]
    start_int = geoInfo.find("[")
    geoInfo = geoInfo[start_int:]
    return simplejson.loads(geoInfo)


def getInfo(data):
    geoData = data
    dataInfo = geoData[15]
    arrDim = dataInfo[0][0][7]["res"]
    dimX, dimY, dimZ = arrDim[0], arrDim[1], arrDim[2]
    geoTypes = geoData[13][5][1][1][5]
    return geoTypes, int(dimX), int(dimY), int(dimZ)


def printBy16(tile):
    for st in range(len(tile) // 16 - 1):
        print st, tile[st * 16:(st + 1) * 16]


def getTileData(tile, mat, xOff, yOff, zOff, comp, dimX, dimY, dimZ):
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
    # make np array of data
    arr = np.array(tile)
    arr2 = np.array(tile)
    # make ends based on previus calculated length and starts
    x_end = x_start + xLen
    y_end = y_start + yLen
    z_end = z_start + zLen
    # if values are all the same AKA compressed
    if (comp == 2):
        arr = np.full((xLen, yLen, zLen), tile)  # Both of these methods work
        arr2 = np.full(xLen * yLen * zLen, tile)  # Both of these methods work
    # method one
    arrMat = arr.reshape((xLen, yLen, zLen))
    # method two
    arrLol = np.empty((xLen, yLen, zLen))
    count = 0
    for x in range(xLen):
        for y in range(yLen):
            for z in range(zLen):
                arrLol[x, y, z] = arr2[count]
                count += 1

    mat[x_start:x_end, y_start:y_end, z_start:z_end] = arrLol
    # mat[x_start:x_end,y_start:y_end,z_start:z_end] = arrMat


def getVoxelDataAt(data, x, y, z, dimX, dimY, dimZ):
    comprArr = data[2][1][3]
    tiles = data[2][1][5]
    mat = np.empty((x, y, z))
    tileCount = 0
    for tx in range((dimX + 15) // 16):
        for ty in range((dimY + 15) // 16):
            for tz in range((dimZ + 15) // 16):
                tile = tiles[tileCount]
                getTileData(tile[3], mat, tx, ty, tz, tile[1], dimX, dimY, dimZ)
                tileCount += 1
    return mat


def getAllVoxelData(geoData, geoInfo, dimX, dimY, dimZ):
    allVoxelData = geoData[19]
    voxelData = {}
    for i in range(len(geoInfo)):
        voxelData[geoInfo[i]] = getVoxelDataAt(allVoxelData[i * 2 + 1], dimX, dimY, dimZ, dimX, dimY, dimZ)
    return voxelData


def geoIntrinsics(data, fields):
    intrinsics = []
    for field in fields:
        voxels = data[field]
        intrinsics.append(voxels.max())
        intrinsics.append(voxels.mean())
        intrinsics.append(np.percentile(voxels, 25))
        intrinsics.append(np.percentile(voxels, 75))
    return intrinsics


def getIntrinsics(filename):
    geoData = pullGeo(filename)
    geoInfo, dimX, dimY, dimZ = getInfo(geoData)
    voxelData = getAllVoxelData(geoData, geoInfo, dimX, dimY, dimZ)
    geoStats = geoIntrinsics(voxelData, ["density", "heat", "temperature"])

    """
    lights = getLights(filename)
    for i in range(57):
        if i < len(lights):
            geoStats.append(lights[i])
        else:
            geoStats.append(0)
    """
    return geoStats


def getLights(filename):
    f = open(filename, "r")
    file_str = f.read()
    f.close()
    l1 = file_str.find("ray_start light")
    string_short = file_str[l1:]
    chunks = []
    # get chuncks AKA diff lights
    l1 = string_short.find("ray_start light")
    while 0 <= l1:
        l2 = string_short.find("ray_end")
        chunks.append(string_short[l1:l2])
        string_short = string_short[l2 + 10:]
        l1 = string_short.find("ray_start light")

    lights = []
    for i, ch in enumerate(chunks):
        s_index = ch.find("lightcolor")
        e_index = ch.find("\n", s_index)
        new_string = ch[s_index:e_index]
        rgb = new_string.split()
        for j in range(1,4):
            lights.append(float(rgb[j]) / 1040.0)
        words = ch.split()
        for j in range(5,21):
            lights.append(words[j])
    return lights


def getCamera(filename):
    f = open(filename, "r")
    file_string = f.read()
    f.close()
    s_index = file_string.find("ray_transform")
    e_index = file_string.find("\n", s_index)
    new_string = file_string[s_index + 13:e_index]
    size = 4
    transform = np.empty([4, 4])
    for i, word in enumerate(new_string.split()):
        transform[i % 4][i // 4] = int(word)
    return transform


if __name__ == '__main__':
    print getIntrinsics("./ifds/fire_lit.ifd")
    lights = getLights("./ifds/fire_lit.ifd")
    print lights
