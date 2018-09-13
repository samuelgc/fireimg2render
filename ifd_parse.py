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
    # dataInfo = geoData[15]
    # arrDim = dataInfo[0][0][7]["res"]
    # dimX, dimY, dimZ = arrDim
    geoTypes = geoData[13][5][1][1][5]
    return geoTypes


def getAllVoxelData(geoData, geoInfo):
    allVoxelData = geoData[19]
    voxelData = {}
    for i in range(len(geoInfo)):
        values = []
        for each in allVoxelData[i * 2 + 1][2][1][5]:
            if isinstance(each[3], list):
                for val in each[3]:
                    values.append(val)
        voxelData[geoInfo[i]] = values
    return voxelData


def geoIntrinsics(data, fields):
    intrinsics = []
    for field in fields:
        voxels = np.asarray(data[field])
        intrinsics.append(voxels.max())
        intrinsics.append(voxels.mean())
        intrinsics.append(np.percentile(voxels, 25))
        intrinsics.append(np.percentile(voxels, 75))
    return intrinsics


def getIntrinsics(filename):
    geoData = pullGeo(filename)
    geoInfo = getInfo(geoData)
    voxelData = getAllVoxelData(geoData, geoInfo)
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
    print getIntrinsics("./ifds/fire_1.ifd")
    #lights = getLights("./ifds/fire_lit.ifd")
    #print lights
