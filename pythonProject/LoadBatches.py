import numpy as np
import cv2
import glob
import itertools
import matplotlib.pyplot as plt
import random
import csv
import pandas as pd
from keras.utils import  np_utils
from PIL import Image

def getImageArr(im):

    img = im.astype(np.float32)

    img[:, :, 0] -= 103.939
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 123.68

    return img

def write_csv_file(path, head, data):
    try:
        with open(path, 'w') as csv_file:
            writer = csv.writer(csv_file, dialect='excel')

            if head is not None:
                writer.writerow(head)

            for row in data:
                writer.writerow(row)

            print("Write a CSV file to path %s Successful." % path)
    except Exception as e:
        print("Write an CSV file to path: %s, Case: %s" % (path, e))
class_values = [0, 16, 32, 48, 64, 80, 96, 112, 128]


def getSegmentationArr(seg, nClasses, input_height, input_width):

    seg_labels = np.zeros((input_height, input_width, nClasses))
    class_values = [0, 16, 32, 48, 64, 80, 96, 112, 128]
    # for i_num in class_values:
    #for c in range(nClasses):  # class_values:#
    #    seg[:, :, 0] += ((seg[:, :, 0] == class_values[c])
    #                         * (colors[c][0])).astype('uint8')
    #    seg[:, :, 1] += ((seg[:, :, 1] == class_values[c])
    #                         * (colors[c][1])).astype('uint8')
    #    seg[:, :, 2] += ((seg[:, :, 2] == class_values[c])
    #                         * (colors[c][2])).astype('uint8')
    #    write_csv_file('C:\\Users\\lenovo\\Desktop\\test_csv\\%d.csv' % c, None, seg_labels[:, :, c])
    for c in range(nClasses):
        if c==0:

            seg_arr_c1 =seg[:, :, 0] != class_values[c]#, seg[:, :, 1] == class_values[c] , seg[:, :, 2] == class_values[c]])
            seg_labels[:, :, c] = seg_arr_c1.astype(int)

            seg_arr_c2 = seg[:, :, 1] != class_values[c]
            seg_labels[:, :, c] =seg_labels[:, :, c] + seg_arr_c2.astype(int)

            seg_arr_c3 = seg[:, :, 2] != class_values[c]
            seg_labels[:, :, c] =seg_labels[:, :, c]+ seg_arr_c3.astype(int)

            seg_arr_correct=seg_labels[:, :, c] <= 0
            seg_labels[:, :, c]=seg_arr_correct.astype(int)
        elif c==1:
            seg_arr_c1 =seg[:, :, 0] != class_values[0]#, seg[:, :, 1] == class_values[c] , seg[:, :, 2] == class_values[c]])
            seg_labels[:, :, c] = seg_arr_c1.astype(int)

            seg_arr_c2 = seg[:, :, 1] != class_values[0]
            seg_labels[:, :, c] =seg_labels[:, :, c] + seg_arr_c2.astype(int)

            seg_arr_c3 = seg[:, :, 2] != class_values[0]
            seg_labels[:, :, c] =seg_labels[:, :, c]+ seg_arr_c3.astype(int)

            seg_arr_correct=seg_labels[:, :, c] > 0
            seg_labels[:, :, c]=seg_arr_correct.astype(int)
        elif c==2:
            seg_arr_c1 =seg[:, :, 0] == class_values[0]#, seg[:, :, 1] == class_values[c] , seg[:, :, 2] == class_values[c]])
            seg_labels[:, :, c] = seg_arr_c1.astype(int)

            seg_arr_c2 = seg[:, :, 1] == class_values[0]
            seg_labels[:, :, c] =seg_labels[:, :, c] + seg_arr_c2.astype(int)

            seg_arr_c3 = seg[:, :, 2] == class_values[8]
            seg_labels[:, :, c] =seg_labels[:, :, c]+ seg_arr_c3.astype(int)

            seg_arr_correct=seg_labels[:, :, c] == 3
            seg_labels[:, :, c]=seg_arr_correct.astype(int)
        elif c == 3:
            seg_arr_c1 = seg[:, :, 0] == class_values[0]  # , seg[:, :, 1] == class_values[c] , seg[:, :, 2] == class_values[c]])
            seg_labels[:, :, c] = seg_arr_c1.astype(int)

            seg_arr_c2 = seg[:, :, 1] == class_values[8]
            seg_labels[:, :, c] = seg_labels[:, :, c] + seg_arr_c2.astype(int)

            seg_arr_c3 = seg[:, :, 2] == class_values[0]
            seg_labels[:, :, c] = seg_labels[:, :, c] + seg_arr_c3.astype(int)

            seg_arr_correct = seg_labels[:, :, c] == 3
            seg_labels[:, :, c] = seg_arr_correct.astype(int)
        elif c == 4:
            seg_arr_c1 = seg[:, :, 0] == class_values[0]  # , seg[:, :, 1] == class_values[c] , seg[:, :, 2] == class_values[c]])
            seg_labels[:, :, c] = seg_arr_c1.astype(int)

            seg_arr_c2 = seg[:, :, 1] == class_values[8]
            seg_labels[:, :, c] = seg_labels[:, :, c] + seg_arr_c2.astype(int)

            seg_arr_c3 = seg[:, :, 2] == class_values[8]
            seg_labels[:, :, c] = seg_labels[:, :, c] + seg_arr_c3.astype(int)

            seg_arr_correct = seg_labels[:, :, c] == 3
            seg_labels[:, :, c] = seg_arr_correct.astype(int)
        elif c == 5:
            seg_arr_c1 = seg[:, :, 0] == class_values[8]  # , seg[:, :, 1] == class_values[c] , seg[:, :, 2] == class_values[c]])
            seg_labels[:, :, c] = seg_arr_c1.astype(int)

            seg_arr_c2 = seg[:, :, 1] == class_values[0]
            seg_labels[:, :, c] = seg_labels[:, :, c] + seg_arr_c2.astype(int)

            seg_arr_c3 = seg[:, :, 2] == class_values[0]
            seg_labels[:, :, c] = seg_labels[:, :, c] + seg_arr_c3.astype(int)

            seg_arr_correct = seg_labels[:, :, c] == 3
            seg_labels[:, :, c] = seg_arr_correct.astype(int)
        elif c == 6:
            seg_arr_c1 = seg[:, :, 0] == class_values[8]  # , seg[:, :, 1] == class_values[c] , seg[:, :, 2] == class_values[c]])
            seg_labels[:, :, c] = seg_arr_c1.astype(int)

            seg_arr_c2 = seg[:, :, 1] == class_values[0]
            seg_labels[:, :, c] = seg_labels[:, :, c] + seg_arr_c2.astype(int)

            seg_arr_c3 = seg[:, :, 2] == class_values[8]
            seg_labels[:, :, c] = seg_labels[:, :, c] + seg_arr_c3.astype(int)

            seg_arr_correct = seg_labels[:, :, c] == 3
            seg_labels[:, :, c] = seg_arr_correct.astype(int)
        elif c == 7:
            seg_arr_c1 = seg[:, :, 0] == class_values[8]  # , seg[:, :, 1] == class_values[c] , seg[:, :, 2] == class_values[c]])
            seg_labels[:, :, c] = seg_arr_c1.astype(int)

            seg_arr_c2 = seg[:, :, 1] == class_values[8]
            seg_labels[:, :, c] = seg_labels[:, :, c] + seg_arr_c2.astype(int)

            seg_arr_c3 = seg[:, :, 2] == class_values[0]
            seg_labels[:, :, c] = seg_labels[:, :, c] + seg_arr_c3.astype(int)

            seg_arr_correct = seg_labels[:, :, c] == 3
            seg_labels[:, :, c] = seg_arr_correct.astype(int)
        elif c == 8:
            seg_arr_c1 = seg[:, :, 0] == class_values[8]  # , seg[:, :, 1] == class_values[c] , seg[:, :, 2] == class_values[c]])
            seg_labels[:, :, c] = seg_arr_c1.astype(int)

            seg_arr_c2 = seg[:, :, 1] == class_values[8]
            seg_labels[:, :, c] = seg_labels[:, :, c] + seg_arr_c2.astype(int)

            seg_arr_c3 = seg[:, :, 2] == class_values[8]
            seg_labels[:, :, c] = seg_labels[:, :, c] + seg_arr_c3.astype(int)

            seg_arr_correct = seg_labels[:, :, c] == 3
            seg_labels[:, :, c] = seg_arr_correct.astype(int)

        #write_csv_file('C:\\Users\\lenovo\\Desktop\\test_csv\\%d.csv' %c, None, seg_labels[:, :, c])


            #seg_labels[:, :, c] = (seg == class_values[c]).astype(int)

    seg_labels = np.reshape(seg_labels, (-1, nClasses))
    return seg_labels


def imageSegmentationGenerator(images_path, segs_path, batch_size,
                               n_classes, input_height, input_width):

    assert images_path[-1] == '/'
    assert segs_path[-1] == '/'

    images = sorted(glob.glob(images_path + "*.jpg") +
                    glob.glob(images_path + "*.png") + glob.glob(images_path + "*.jpeg"))

    segmentations = sorted(glob.glob(segs_path + "*.jpg") +
                           glob.glob(segs_path + "*.png") + glob.glob(segs_path + "*.jpeg"))

    zipped = itertools.cycle(zip(images, segmentations))

    while True:
        X = []
        Y = []
        for _ in range(batch_size):
            im, seg = zipped.__next__()
            im = cv2.imread(im, 1)
            seg = cv2.imread(seg, 1)#0
            #segr,segg,segb=cv2.split(seg)
            #plt.imshow(segr)
            #plt.show()
            #plt.imshow(segg)
            #plt.show()
            #plt.imshow(segb)
            #plt.show()

            #write_csv_file('C:\\Users\\lenovo\\Desktop\\test_csv\\r.csv' , None, segr[:, :])
            #write_csv_file('C:\\Users\\lenovo\\Desktop\\test_csv\\g.csv' , None, segg[:, :])
            #write_csv_file('C:\\Users\\lenovo\\Desktop\\test_csv\\b.csv' , None, segb[:, :])
            #write_csv_file('C:\\Users\\lenovo\\Desktop\\test_csv\\%d.csv' % c, None, seg_labels[:, :, c])


            #print("seg_shape:",seg.shape)
            #print("seg_num:",np.unique(seg))

            assert im.shape[:2] == seg.shape[:2]

            assert im.shape[0] >= input_height and im.shape[1] >= input_width

            #xx = random.randint(0, im.shape[0] - input_height)
            #yy = random.randint(0, im.shape[1] - input_width)

            #im = im[xx:xx + input_height, yy:yy + input_width]
            #seg = seg[xx:xx + input_height, yy:yy + input_width]

            X.append(getImageArr(im))
            Y.append(
                getSegmentationArr(
                    seg,
                    n_classes,
                    input_height,
                    input_width))

        yield np.array(X), np.array(Y)


if __name__ == '__main__':
    G = imageSegmentationGenerator("C:\\Users\\lenovo\\Desktop\\train_input_es/",
                                   "C:\\Users\\lenovo\\Desktop\\train_mask_es/", batch_size=100, n_classes=9, input_height=320, input_width=480)
    x, y = G.__next__()
    print(x.shape, y.shape)
