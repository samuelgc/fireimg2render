import os
import csv
import numpy as np
from PIL import Image
import colorsys


FILE_DIRS = ['extra', 'google/fire', 'google/explosion', 'google/flamethrower', 'google/forest_fire', 'image_net/fireplace', 'image_net/other_fire']
DATA_FILES = ['extra', 'google_fire', 'google_explosion', 'google_flamethrower', 'google_forest_fire', 'image_net_fireplace', 'image_net_other']


"""
Creates a mask of fire regions in img
r_t values ranging from 55 to 65
s_t values ranging from 115 to 135
"""
def mask(img, r_t=60, s_t=125):
    wid, hi = img.size
    ycc = img.convert('YCbCr')
    # hsv = img.convert('HSV')

    out = Image.new('1', (wid, hi))
    out_pixels = out.load()

    for i in range(wid):
        for j in range(hi):
            r, g, b = img.getpixel((i, j))
            y, c_b, c_r = ycc.getpixel((i, j))
            # h, s, v = hsv.getpixel((i, j))
            h, s, v = colorsys.rgb_to_hsv(r/255., g/255., b/255.)

            r1 = r >= r_t
            r2 = r >= 1.2*g >= 2*b
            r3 = s >= ((255-r) * s_t/r_t)

            r4 = y >= 1.5*c_b
            r5 = c_r >= 1.5*c_b

            if r1 and r2 and r3 or r4 and r5:
                out_pixels[i, j] = 1
    return out


"""
Saves the masked image
"""
def save_mask(img, msk, filename):
    pixels = img.load()
    for i in range(msk.size[0]):
        for j in range(msk.size[1]):
            if msk.getpixel((i, j)) == 0:
                pixels[i, j] = (0, 0, 0)
    img.save('./masks/' + filename, "JPEG")


"""
Calculates the temperature stats for the pixels in img with mask
"""
def map_stats(img, msk):
    stats = []
    fire_pxl = []
    for i in range(msk.size[0]):
        for j in range(msk.size[1]):
            if msk.getpixel((i, j)) == 1:
                r, g, b = img.getpixel((i, j))

                # According to PIL library converts to CIE XYZ color space
                # Other sources also seem to name it as correct for transforming sRGB to CIE XYZ
                tx = 0.412453 * r + 0.357580 * g + 0.180423 * b
                ty = 0.212671 * r + 0.715160 * g + 0.072169 * b
                tz = 0.019334 * r + 0.119193 * g + 0.950227 * b

                """
                # From paper but seems less accurate
                tx = -0.14282*r + 1.54924*g - 0.95641*b
                ty = -0.32466*r + 1.57837*g - 0.73191*b
                tz = -0.68202*r + 0.77073*g + 0.56332*b
                """

                cct = 0
                tsum = tx + ty + tz
                if tsum != 0:
                    nx = tx / tsum
                    ny = ty / tsum

                    n = (nx - 0.3320) / (0.1858 - ny)
                    cct = 449 * n ** 3 + 3525 * n ** 2 + 6823.3 * n + 5520.33

                fire_pxl.append(cct)

    if len(fire_pxl) != 0:
        stats.append(np.mean(fire_pxl))             # Mean
        stats.append(np.var(fire_pxl))              # Variation
        stats.append(np.std(fire_pxl))              # Standard Deviation
        stats.append(np.amin(fire_pxl))             # Min value
        stats.append(np.percentile(fire_pxl, 25))   # 25th percentile
        stats.append(np.median(fire_pxl))           # Median
        stats.append(np.percentile(fire_pxl, 75))   # 75th percentile
        stats.append(np.amax(fire_pxl))             # Max value
        stats.append(np.ptp(fire_pxl))              # Range
    return stats


"""
Writes a csv file of stats
"""
def write_file(file, stats):
    with open('./train_data/'+file, "w+") as f:
        writer = csv.writer(f)
        writer.writerows(stats)
        f.close()


"""
Write normalized data to csv 
Normalizes temperature values between 0 and 25000
"""
def normalize_data():
    file_dir = './train_data/'
    for filename in os.listdir(file_dir):
        if os.path.isfile(filename):
            data = np.loadtxt(file_dir + filename, delimiter=",")
            data /= 25000
            with open(file_dir + 'normalized/' + filename, "w+") as f:
                writer = csv.writer(f)
                writer.writerows(data)
                f.close()


def main():
    for i in range(len(FILE_DIRS)):
        file_dir = './fire_images/' + FILE_DIRS[i]
        img_stats = []
        for filename in os.listdir(file_dir):
            fire_img = Image.open(file_dir + '/' + filename)
            fire_mask = mask(fire_img)
            fire_stats = map_stats(fire_img, fire_mask)
            if len(fire_stats) != 0:
                img_stats.append(fire_stats)
        write_file(DATA_FILES[i] + '.csv', img_stats)
        print "Finished file: {}".format(DATA_FILES[i])


if __name__== '__main__':
    main()