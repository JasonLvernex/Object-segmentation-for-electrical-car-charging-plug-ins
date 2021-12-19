# -*- coding: utf-8 -*-
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.utils import  np_utils
from keras.models import Sequential
from keras.layers import Dense,Activation,Conv2D,MaxPooling2D,Flatten,Dropout,Input,Reshape,UpSampling2D,concatenate
from keras.optimizers import  Adam
#from Keras import argm
from sklearn.model_selection import KFold
from sklearn import datasets, svm, metrics
from keras.utils.vis_utils import plot_model
from keras.utils import multi_gpu_model
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
import LoadBatches
import cv2
import six
import random
import csv
import math
import tensorflow as tf
import keras.backend as K
from keras.losses import binary_crossentropy
from sklearn.utils import class_weight

#import tqdm
#from keras import backend as K

#K.tensorflow_backend._get_available_gpus()
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())
import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "/device:GPU:0"

#from skimage import io
#img=io.imread("mask.png", as_grey=False)
#io.imshow(img)
class_colors = [(random.randint(0, 255), random.randint(
    0, 255), random.randint(0, 255)) for _ in range(5000)]

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

train_images_dir = "C:\\Users\\lenovo\\Desktop\\new_origin/"
train_annotations_dir = "C:\\Users\\lenovo\\Desktop\\new_mask/"
train_batch_size=100
n_classes=9
input_height=320
input_width=480
num=1
out_path1="C:\\Users\\lenovo\\Desktop\\train_input_es/"
out_path2="C:\\Users\\lenovo\\Desktop\\train_mask_es/"
#original 1920*1080

##this part is used
image_train_reshape_trigger=0
if image_train_reshape_trigger==1:
 input("testing:")
 for item in os.listdir(train_images_dir):
    img = cv2.imread(train_images_dir + "\\img%d.png" % num)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    crop = img[300:940, 400:1360]  # 这里需要注意[ymin:ymax, xmin:xmax]
    demand = cv2.resize(crop, (480, 320))
    #plt.imshow(demand)
    plt.show()
    img = cv2.imwrite(out_path1 + "\\img%d.png" % num, demand)
    num+=1
image_mask_reshape_trigger=0
if image_mask_reshape_trigger==1:
 num=1
 for item in os.listdir(train_annotations_dir):
    img = cv2.imread(train_annotations_dir + "\\label%d.png" % num)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    crop = img[300:940, 400:1360]  # 这里需要注意[ymin:ymax, xmin:xmax]
    demand = cv2.resize(crop, (480, 320))
    #plt.imshow(demand)
    plt.show()
    img = cv2.imwrite(out_path2 + "\\label%d.png" % num, demand)
    num+=1
#input("finished,continue?")

#img_size = 320  # 根据实际情况可修改

#nub_train = len(glob(train_path + '/*/*.jpg'))
# 先生成空array，然后往里填每张图片的array
#x = np.zeros((547, img_size, img_size, 3), dtype=np.uint8)
#y = np.zeros((547,img_size, img_size, 3), dtype=np.uint8)
#i = 0
#j = 0

#for img_path in glob(train_images_dir + '/*/*.png'):#os.listdir(train_annotations_dir):#glob(train_images_dir + '/*/*.png'): #tqdm(glob(train_path + '/*/*.jpg')):
#    img = Image.open(img_path)
#    img = img.resize((input_height, input_width))  # 图片resize
#    img.save(os.path.join(out_path2, img_path))
    #arr = np.asarray(img)  # 图片转array
    #x[i, :, :, :] = arr  # 赋值
#    i += 1
#for img_path in glob(train_annotations_dir + '/*/*.png'):#os.listdir(train_images_dir):#glob(train_annotations_dir + '/*/*.png'):  # tqdm(glob(train_path + '/*/*.jpg')):
#    img = Image.open(img_path)
#    img = img.resize((input_height, input_width))  # 图片resize
    #arr = np.asarray(img)  # 图片转array
    #y[j, :, :, :] = arr  # 赋值
#    j += 1
G=LoadBatches.imageSegmentationGenerator(out_path1,out_path2, train_batch_size, n_classes=n_classes, input_height=input_height, input_width=input_width)
x, y = G.__next__()
print(x.shape, y.shape)
print(np.unique(y[1]))
write_csv_file('C:\\Users\\lenovo\\Desktop\\y1.csv',None,y[1][6])

print("unique y:",np.unique(y))

def creat_class_weight(labels_dict,mu=0.15):
    total=np.sum(labels_dict.values())
    keys=labels_dict.keys()
    class_weight=dict()

    for key in keys:
        #score=math.log(mu*total/float(labels_dict[key]))
        #if key == 0:
        score=5
        class_weight[key]=score if key >=1 else 1.0

        return class_weight



#img_path = train_images_dir
#img = Image.open(img_path)
#print(img.mode)
#for item in train_images_dir:
#    x.append
def weightedLoss(originalLossFunc, weightsList):

    def lossFunc(true, pred):

        axis = -1 #if channels last
        #axis=  1 #if channels first


        #argmax returns the index of the element with the greatest value
        #done in the class axis, it returns the class index
        classSelectors = K.argmax(true, axis=axis)
        classSelectors = tf.cast(classSelectors, tf.int32)
            #if your loss is sparse, use only true as classSelectors

        #considering weights are ordered by class, for each class
        #true(1) if the class index is equal to the weight index
        classSelectors = [K.equal(i, classSelectors) for i in range(len(weightsList))]

        #casting boolean to float for calculations
        #each tensor in the list contains 1 where ground true class is equal to its index
        #if you sum all these, you will get a tensor full of ones.
        classSelectors = [K.cast(x, K.floatx()) for x in classSelectors]

        #for each of the selections above, multiply their respective weight
        weights = [sel * w for sel,w in zip(classSelectors, weightsList)]

        #sums all the selections
        #result is a tensor with the respective weight for each element in predictions
        weightMultiplier = weights[0]
        for i in range(1, len(weights)):
            weightMultiplier = weightMultiplier + weights[i]


        #make sure your originalLossFunc only collapses the class axis
        #you need the other axes intact to multiply the weights tensor
        loss = originalLossFunc(true,pred)
        loss = loss * weightMultiplier

        return loss
    return lossFunc

#K-fold process
kf=KFold(n_splits=2,shuffle=True)    #分成几个组
#kf.shuffle()
kf.get_n_splits(train_images_dir)
print(kf)
for train_index,test_index in kf.split(x):
    #print("Train Index:",train_index,",Test Index:",test_index)
    X_train,X_test=x[train_index,:],x[test_index,:]
    Y_train,Y_test=y[train_index,:],y[test_index,:]
print(X_train.shape, X_test.shape)
print(Y_train.shape, Y_test.shape)
print(Y_test[1])
print("unique Y1:",np.unique(Y_test[1]))
#y = [0,0,0,0,0,0,0,0,1,1,1,1,1,1,2,2]  #标签值，一共16个样本

#from sklearn.utils.class_weight import compute_class_weight
#class_weight = {0:1,1:3,2:5}   # {class_label_1:weight_1, class_label_2:weight_2, class_label_3:weight_3}
#classes = np.array([0, 1, 2])  #标签类别
#weight = compute_class_weight(class_weight, classes, y)
#print(weight)   # 输出：[1. 3. 5.]，也就是字典中设置的值
class_weightss = {"background":1,"rect":3,1:5,2:5,3:5,4:5,5:5,6:5,7:5}
class_weightsse = {0:1,1:3,2:5,3:5,4:5,5:5,6:5,7:5,8:5}
class_weightee= {0:1.,1:3.}
#class_weightsses = [1,3,5,5,5,5,5,5,5]
#class_weightsses=np.array(class_weightsses)
classes = np.array([[1.,0.,0.,0.,0.,0.,0.,0.,0.],
                    [0.,1.,0.,0.,0.,0.,0.,0.,0.],
                    [0.,0.,1.,0.,0.,0.,0.,0.,0.],
                    [0.,0.,0.,1.,0.,0.,0.,0.,0.],
                    [0.,0.,0.,0.,1.,0.,0.,0.,0.],
                    [0.,0.,0.,0.,0.,1.,0.,0.,0.],
                    [0.,0.,0.,0.,0.,0.,1.,0.,0.],
                    [0.,0.,0.,0.,0.,0.,0.,1.,0.],
                    [0.,0.,0.,0.,0.,0.,0.,0.,1.]])  #标签类别
labels_dict={0:1.,1:3.,2:5.,3:5.,4:5.,5:5.,6:5.,7:5.,8:5.}
aa=creat_class_weight(labels_dict)
#class_weights = class_weight.compute_class_weight(class_weight=class_weightss,
#                                                 classes=classes,
#                                                y= Y_train)
#class_weights = class_weight.compute_class_weight(class_weight='balanced',
#                                                 classes=np.unique(Y_train),
#                                                y= np.array(Y_train))
#data pre-processing
input_height=320
input_width=480
X_train=X_train.reshape(-1,input_height,input_width,3)#-1 是样本个数，当不确定有多少样本的时候就用-1站位,1是黑白照片，28pixal
X_test=X_test.reshape(-1,input_height,input_width,3)
#Y_train=np_utils.to_categorical(Y_train,num_classes=9)
#Y_test=np_utils.to_categorical(Y_test,num_classes=9)
#print(Y_train.shape, Y_test.shape)


img_input = Input(shape=(input_height,input_width , 3 ))
#coding
conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(img_input)
conv1 = Dropout(0.2)(conv1)
conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
pool1 = MaxPooling2D((2, 2))(conv1)

conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
conv2 = Dropout(0.2)(conv2)
conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
pool2 = MaxPooling2D((2, 2))(conv2)

conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
conv3 = Dropout(0.2)(conv3)
conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
#decoding
up1 = concatenate([UpSampling2D((2, 2))(conv3), conv2], axis=-1)
conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
conv4 = Dropout(0.2)(conv4)
conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)

up2 = concatenate([UpSampling2D((2, 2))(conv4), conv1], axis=-1)
conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(up2)
conv5 = Dropout(0.2)(conv5)
conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)

out = Conv2D( n_classes, (1, 1) , padding='same')(conv5)

from keras_segmentation.models.model_utils import get_segmentation_model

model = get_segmentation_model(img_input ,  out ) # this would build the segmentation model
#model=multi_gpu_model(model,gpus="/device:XLA_GPU:0")
print(model.output_shape)

model.summary()
plot_model(model, to_file='model_project.png',show_shapes=True)

#model.add(Reshape((153600, 9)))
##


#from keras_segmentation.models.model_utils import get_segmentation_model

#model = get_segmentation_model(img_input ,  out ) # this would build the segmentation model
#model.add(Dense(8))
#model.add(Activation('softmax'))



#another way to define your optimizer
adam=Adam(lr=1e-4)
#we add metrics to get more results you want to see
weightlist=[0.01,0.1,800,800,800,800,800,800,800]
model.compile(optimizer=adam,loss=weightedLoss(binary_crossentropy,weightlist),metrics=['accuracy'])#loss='poisson''categorical_crossentropy'
print('training----')
#another way to train the model
#X_train.reshape(100,320,480,3)
historys =model.fit(X_train,Y_train,epochs=70,batch_size=10,validation_data=(X_test,Y_test)) #each time 32 datas with two rounds
print('\ntesting-----')
#evaluate the model with metrics we defined earlier

loss,accuracy=model.evaluate(X_test,Y_test)

print('test loss=',loss)
print('test accuracy=',accuracy)

plot_model(model, to_file='model_test.png',show_shapes=True)
model.summary()

def get_colored_segmentation_image(seg_arr, n_classes, colors=class_colors):
    output_height = seg_arr.shape[0]
    output_width = seg_arr.shape[1]
    class_values = [0, 16, 32, 48, 64, 80, 96, 112, 128]

    seg_img = np.zeros((output_height, output_width, 3))

    #for c in range(n_classes):#class_values:#
    #    seg_img[:, :, 0] += ((seg_arr[:, :, 0] == class_values[c])
    #                         * (colors[c][0])).astype('uint8')
    #    seg_img[:, :, 1] += ((seg_arr[:, :, 1] == class_values[c])
    #                         * (colors[c][1])).astype('uint8')
    #    seg_img[:, :, 2] += ((seg_arr[:, :, 2] == class_values[c])
    #                         * (colors[c][2])).astype('uint8')

    for c in range(n_classes):
        seg_arr_c = seg_arr[:, :] ==c#class_values[c] #c
        seg_img[:, :, 0] += ((seg_arr_c)*(colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((seg_arr_c)*(colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((seg_arr_c)*(colors[c][2])).astype('uint8')

    return seg_img

def overlay_seg_image(inp_img, seg_img):
    orininal_h = inp_img.shape[0]
    orininal_w = inp_img.shape[1]
    seg_img = cv2.resize(seg_img, (orininal_w, orininal_h), interpolation=cv2.INTER_NEAREST)

    fused_img = (inp_img/2 + seg_img/2).astype('uint8')
    return fused_img

def get_legends(class_names, colors=class_colors):

    n_classes = len(class_names)
    legend = np.zeros(((len(class_names) * 25) + 25, 125, 3),
                      dtype="uint8") + 255

    class_names_colors = enumerate(zip(class_names[:n_classes],
                                       colors[:n_classes]))

    for (i, (class_name, color)) in class_names_colors:
        color = [int(c) for c in color]
        cv2.putText(legend, class_name, (5, (i * 25) + 17),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
        cv2.rectangle(legend, (100, (i * 25)), (125, (i * 25) + 25),
                      tuple(color), -1)

    return legend

def concat_lenends(seg_img, legend_img):

    new_h = np.maximum(seg_img.shape[0], legend_img.shape[0])
    new_w = seg_img.shape[1] + legend_img.shape[1]

    out_img = np.zeros((new_h, new_w, 3)).astype('uint8') + legend_img[0, 0, 0]

    out_img[:legend_img.shape[0], :  legend_img.shape[1]] = np.copy(legend_img)
    out_img[:seg_img.shape[0], legend_img.shape[1]:] = np.copy(seg_img)

    return out_img

class DataLoaderError(Exception):
    pass

IMAGE_ORDERING_CHANNELS_LAST = "channels_last"
IMAGE_ORDERING_CHANNELS_FIRST = "channels_first"

# Default IMAGE_ORDERING = channels_last
IMAGE_ORDERING = IMAGE_ORDERING_CHANNELS_LAST

def get_image_array(image_input,
                    width, height,
                    imgNorm="sub_mean", ordering='channels_first', read_image_type=1):
    """ Load image array from input """

    if type(image_input) is np.ndarray:
        # It is already an array, use it as it is
        img = image_input
    elif isinstance(image_input, six.string_types):
        if not os.path.isfile(image_input):
            raise DataLoaderError("get_image_array: path {0} doesn't exist"
                                  .format(image_input))
        img = cv2.imread(image_input, read_image_type)
    else:
        raise DataLoaderError("get_image_array: Can't process input type {0}"
                              .format(str(type(image_input))))

    if imgNorm == "sub_and_divide":
        img = np.float32(cv2.resize(img, (width, height))) / 127.5 - 1
    elif imgNorm == "sub_mean":
        img = cv2.resize(img, (width, height))
        img = img.astype(np.float32)
        img = np.atleast_3d(img)

        means = [103.939, 116.779, 123.68]

        for i in range(min(img.shape[2], len(means))):
            img[:, :, i] -= means[i]

        img = img[:, :, ::-1]
    elif imgNorm == "divide":
        img = cv2.resize(img, (width, height))
        img = img.astype(np.float32)
        img = img/255.0

    if ordering == 'channels_first':
        img = np.rollaxis(img, 2, 0)
    return img


nummer=1
def predict( inp=None, out_fname=None,
             overlay_img=False,
            class_names=None, show_legends=False, colors=class_colors,
            prediction_width=None, prediction_height=None,innum=nummer):


    output_width = model.output_width
    output_height = model.output_height
    input_width = model.input_width
    input_height = model.input_height
    n_classes = 9#model.n_classes


    inp = Image.open(inp)
    inp = inp.resize((input_width,input_height))  # 图片resize
    inp = np.float32(inp)
    inp=cv2.resize(inp, (input_width, input_height))
    #print("inp_shape:",inp.shape)

    #x = np.asarray(inp)  # 图片转array
    x = get_image_array(inp, input_width, input_height,
                        ordering=IMAGE_ORDERING)
    print(x.shape)
    pr = model.predict(np.array([x]))#[0]#np.array([x])
    pr = pr.reshape((output_height,  output_width, n_classes)).argmax(axis=2)
    print("pr:", pr.shape)
    for jet in range(0,9):
        write_csv_file('C:\\Users\\lenovo\\Desktop\\testout_csv\\%d.csv' % jet, None, pr[:, :])

    print("pr_num",np.unique(pr))

    #seg_img = visualize_segmentation(pr, inp, n_classes=n_classes,
    #                                 colors=colors, overlay_img=overlay_img,
    #                                 show_legends=show_legends,
    #                                 class_names=class_names,
    #                                 prediction_width=prediction_width,
    #                                 prediction_height=prediction_height)
    if n_classes is None:
        n_classes = np.max(pr)

    seg_img = get_colored_segmentation_image(pr, n_classes, colors=colors)

    if inp is not None:
        original_h = input_height#inp.shape[0]
        original_w = input_width#inp.shape[1]
        seg_img = cv2.resize(seg_img, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

    if (prediction_height is not None) and (prediction_width is not None):
        seg_img = cv2.resize(seg_img, (prediction_width, prediction_height), interpolation=cv2.INTER_NEAREST)
        if inp is not None:
            inp = np.float32(inp)
            inp = cv2.resize(inp,(prediction_width, prediction_height))

    if overlay_img:
        assert inp is not None
        seg_img = overlay_seg_image(inp, seg_img)

    if show_legends:
        assert class_names is not None
        legend_img = get_legends(class_names, colors=colors)

        seg_img = concat_lenends(seg_img, legend_img)

    if out_fname is not None:
        cv2.imwrite(out_fname+"predict%d.png" %innum, seg_img)
        #plt.imshow(kk)
    #input("satisfied?:")
    return pr
class_name=["background","rect","1","2","3","4","5","6","7"]
#predict( inp="C:\\Users\\lenovo\\Desktop\\train_input_es\\img215\\.png", out_fname="C:\\Users\\lenovo\\Desktop\\predict_test",
#            overlay_img=False,
#            class_names=class_name, show_legends=True, colors=class_colors,
#            prediction_width=480, prediction_height=320)
ACCEPTABLE_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".bmp"]
ACCEPTABLE_SEGMENTATION_FORMATS = [".png", ".bmp"]
def get_pairs_from_paths(images_path, segs_path, ignore_non_matching=False, other_inputs_paths=None):
    """ Find all the images from the images_path directory and
        the segmentation images from the segs_path directory
        while checking integrity of data """



    image_files = []
    segmentation_files = {}

    for dir_entry in os.listdir(images_path):
        if os.path.isfile(os.path.join(images_path, dir_entry)) and \
                os.path.splitext(dir_entry)[1] in ACCEPTABLE_IMAGE_FORMATS:
            file_name, file_extension = os.path.splitext(dir_entry)
            image_files.append((file_name, file_extension,
                                os.path.join(images_path, dir_entry)))

    if other_inputs_paths is not None:
        other_inputs_files = []

        for i, other_inputs_path in enumerate(other_inputs_paths):
            temp = []

            for y, dir_entry in enumerate(os.listdir(other_inputs_path)):
                if os.path.isfile(os.path.join(other_inputs_path, dir_entry)) and \
                        os.path.splitext(dir_entry)[1] in ACCEPTABLE_IMAGE_FORMATS:
                    file_name, file_extension = os.path.splitext(dir_entry)

                    temp.append((file_name, file_extension,
                                 os.path.join(other_inputs_path, dir_entry)))

            other_inputs_files.append(temp)

    for dir_entry in os.listdir(segs_path):
        if os.path.isfile(os.path.join(segs_path, dir_entry)) and \
           os.path.splitext(dir_entry)[1] in ACCEPTABLE_SEGMENTATION_FORMATS:
            file_name, file_extension = os.path.splitext(dir_entry)
            full_dir_entry = os.path.join(segs_path, dir_entry)
            if file_name in segmentation_files:
                raise DataLoaderError("Segmentation file with filename {0}"
                                      " already exists and is ambiguous to"
                                      " resolve with path {1}."
                                      " Please remove or rename the latter."
                                      .format(file_name, full_dir_entry))

            segmentation_files[file_name] = (file_extension, full_dir_entry)

    return_value = []
    # Match the images and segmentations
    for image_file, _, image_full_path in image_files:
        if image_file in segmentation_files:
            if other_inputs_paths is not None:
                other_inputs = []
                for file_paths in other_inputs_files:
                    success = False

                    for (other_file, _, other_full_path) in file_paths:
                        if image_file == other_file:
                            other_inputs.append(other_full_path)
                            success = True
                            break

                    if not success:
                        raise ValueError("There was no matching other input to", image_file, "in directory")

                return_value.append((image_full_path,
                                     segmentation_files[image_file][1], other_inputs))
            else:
                return_value.append((image_full_path,
                                     segmentation_files[image_file][1]))
        elif ignore_non_matching:
            continue
        else:
            # Error out
            raise DataLoaderError("No corresponding segmentation "
                                  "found for image {0}."
                                  .format(image_full_path))

    return return_value

def _get_colored_segmentation_image(img, seg, colors,
                                    n_classes):
    """ Return a colored segmented image """
    seg_img = np.zeros_like(seg)
    class_values=[0,16,32,48,64,80,96,112,128]

    #i_count=0
    for c in range(n_classes):#class_values:#
        seg_img[:, :, 0] += ((seg[:, :, 0] == class_values[c])
                             * (colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((seg[:, :, 1] == class_values[c])
                             * (colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((seg[:, :, 2] == class_values[c])
                             * (colors[c][2])).astype('uint8')
        #i_count+=1

    return img, seg_img

def visualize_segmentation_dataset_one(images_path, segs_path, n_classes,
                                        no_show=False,ignore_non_matching=False):

    #img_seg_pairs = get_pairs_from_paths(
    #                            images_path, segs_path,
    #                            ignore_non_matching=ignore_non_matching)

    colors = class_colors

    #im_fn, seg_fn = random.choice(img_seg_pairs)

    img = cv2.imread(images_path)
    seg = cv2.imread(segs_path)
    print("Found the following classes "
          "in the segmentation image:", np.unique(seg))

    img, seg_img = _get_colored_segmentation_image(
                                        img, seg, colors,
                                        n_classes)

    if not no_show:
        cv2.imshow("img", img)
        cv2.imshow("seg_img", seg_img)
        cv2.waitKey()

    return img, seg_img
#############################################
##visualize_segmentation_dataset_one("C:\\Users\\lenovo\\Desktop\\train_input_es\\img28.png", "C:\\Users\\lenovo\\Desktop\\train_mask_es\\label28.png", 9,no_show=False,ignore_non_matching=False)

#paths = get_pairs_from_paths(out_path1, out_path2)
#paths = list(zip(*paths))
paths="C:\\Users\\lenovo\\Desktop\\train_input_es\\"
#paths = list(zip(*paths))
#inp_images = list(paths[0])
#annotations = list(paths[1])

#assert type(inp_images) is list
#assert type(annotations) is list

tp = np.zeros(model.n_classes)
fp = np.zeros(model.n_classes)
fn = np.zeros(model.n_classes)
n_pixels = np.zeros(model.n_classes)

#for inp, ann in tqdm(zip(inp_images, annotations)):
#for inp in zip(inp_images):
#    pr = predict( inp,  overlay_img=False,out_fname="C:\\Users\\lenovo\\Desktop\\predict_test",
#            class_names=class_name, show_legends=True, colors=class_colors,
#            prediction_width=480, prediction_height=320)
nummer=1
for filename in os.listdir(paths):
   pr = predict(paths+filename, overlay_img=False, out_fname="C:\\Users\\lenovo\\Desktop\\predict_test\\",
             class_names=class_name, show_legends=True, colors=class_colors,
             prediction_width=480, prediction_height=320,innum=nummer)
   nummer+=1
    #gt = get_segmentation_array(ann, model.n_classes,
    #                            model.output_width, model.output_height,
    #                            no_reshape=True, read_image_type=read_image_type)
#Y_test=Y_test.reshape(-1,input_height,input_width,9)
#print(Y_test.shape)
#print(Y_test[1])
print(historys.history.keys())
def figure(historys):
    acc = historys.history['accuracy']
    val_acc = historys.history['val_accuracy']
    loss = historys.history['loss']
    val_loss = historys.history['val_loss']
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='test accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig('Training and test accuracy_project.png')
    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='test loss')
    plt.title('Training and test loss')
    plt.legend()
    plt.savefig('Training and test loss_project')
    plt.show()

figure(historys)