from model import Resnet
from processing import data_generate
from result import predict, result_show
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras import models
import numpy as np
from math import ceil

import os
# # 防止输出提示
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# # 使用 GPU 0，1
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

filenames = [
    "breastmnist",
    "chestmnist",
    "dermamnist",
    "octmnist",
    "organmnist_axial",
    "organmnist_coronal",
    "organmnist_sagittal",
    "pathmnist",
    "pneumoniamnist",
    "retinamnist"
]
task_types = [
    ['binary-class'],
    ['multi-label', 'binary-class'],
    ['multi-class'],
    ['multi-class'],
    ['multi-class'],
    ['multi-class'],
    ['multi-class'],
    ['multi-class'],
    ['binary-class'],
    ['ordinal regression']
]
print("输入想要处理的数据集的序号：")
for i in range(len(filenames)):
    print("%d:" % i, filenames[i])
option = int(input())
# option = 1
data_file = filenames[option]
task_type = task_types[option]

print("输入网络类型：\n0:ResNet 18\n1:ResNet 50")
option = int(input())
# option = 1
net_type = 18 if option == 0 else 50

use_saved_model = bool(int(input("是否使用已有的模型：\n0:否\n1:是\n")))
# use_saved_model = 0

data = np.load("./code/medmnist/dataset/" + data_file + ".npz")

if use_saved_model:
    model = models.load_model(
        "./hw/myoutput/" + data_file + "/ResNet%d_" % net_type + data_file + ".h5")
else:
    if len(task_type) == 1:
        train_labels = to_categorical(data["train_labels"])
        val_labels = to_categorical(data["val_labels"])
    else:
        train_labels = data["train_labels"]
        val_labels = data["val_labels"]

    W = 1 if len(data["train_images"].shape) == 3 else 3
    classes = len(train_labels[0])
    if net_type == 18:
        model = Resnet().resnet_18(W, classes, task_type)
    else:
        model = Resnet().resnet_50(W, classes, task_type)

    if len(task_type) == 1:
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
    else:
        model.compile(loss='binary_crossentropy',
                      optimizer='adam', metrics=['accuracy'])

    if net_type == 18:
        batch_size = 16
    else:
        batch_size = 4

    epoches = 20
    checkpoint = ModelCheckpoint(filepath="./hw/myoutput/" + data_file + "/ResNet%d_" % net_type + data_file + ".h5",
                                 monitor='val_accuracy',
                                 save_best_only='True',
                                 mode='auto',
                                 period=1)
    history = model.fit_generator(data_generate(data['train_images'], train_labels, batch_size),
                                  steps_per_epoch=ceil(
                                      len(train_labels) / batch_size),
                                  epochs=epoches,
                                  validation_data=data_generate(
                                      data['val_images'], val_labels, batch_size),
                                  validation_steps=ceil(
                                      len(val_labels) / batch_size),
                                  callbacks=[checkpoint])
    model = models.load_model(
        "./hw/myoutput/" + data_file + "/ResNet%d_" % net_type + data_file + ".h5")
    # model.save("./hw/myoutput/" + data_file + "/ResNet%d_"%net_type + data_file + ".h5")
    result_show(history, data_file, net_type)

if len(task_type) == 1:
    test_labels = to_categorical(data["test_labels"])
else:
    test_labels = data["test_labels"]
predict(model, data, test_labels, data_file, net_type)
