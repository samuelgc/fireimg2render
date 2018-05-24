from PIL import Image
import urllib2
import os

from mask_fire import *

init_file = '/home/samuelgc/Documents/Datasets/fire_img_urls.txt'
out_dir = './fire_images/'
size = 256, 256


def scrape_urls():

    with open(init_file) as f:
        content = [x.strip('\n') for x in f.readlines()]

        i = 0
        for url in content:
            try:
                img = Image.open(urllib2.urlopen(url))
                img.save('/home/samuelgc/Documents/Datasets/fire_imgs/{}.jpg'.format(i), format="JPEG")
                i += 1
                print "Saved Image {}".format(i)
            except Exception as e:
                print("Could not save : ", img)


def thumbnails(dir):
    for filename in os.listdir(dir):
        try:
            img = Image.open(dir + '/' + filename)
            img.thumbnail(size)
            img.save(out_dir + 'image_net/fire/' + filename, "JPEG")
        except IOError:
            print("Could not create thumbnail for", filename)


def main():
    file_dir = '/home/samuelgc/Documents/Datasets/fire_imgs/fire'
    for filename in os.listdir(file_dir):
        fire_img = Image.open(file_dir + '/' + filename)
        fire_mask = mask(fire_img)
        save_mask(fire_img, fire_mask, filename)
    #thumbnails(file_dir)
    #scrape_urls()


if __name__ == '__main__':
    main()