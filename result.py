import matplotlib.pyplot as plt
from processing import test_data_generate
from math import ceil
import pandas as pd
from numpy import argmax, concatenate

def predict(model, data, test_labels, data_file, net_type):
    batch_size = 8
    ans = model.predict_generator(test_data_generate(
        data['test_images'], test_labels, batch_size),
        steps=ceil(len(test_labels) / batch_size))

    count = 0
    if len(data['test_labels'][0]) > 1: #表明这是一个多标签问题
        for i in range(len(ans)):
            for j in range(len(ans[0])):
                if data["test_labels"][i][j] == round(ans[i][j]):
                    count += 1
        acc = count / (len(ans) * len(ans[0]))
    else:
        for i in range(len(ans)):
            if data["test_labels"][i][0] == argmax(ans[i]):
                count += 1
        acc = count / len(ans)

    print("test acc: ", acc)
    file_handle=open("./hw/myoutput/" + data_file + "/ResNet%d_"%net_type + data_file + ".txt", mode='w')
    file_handle.write(str(acc))
    file_handle.close()

    name = []
    if len(data['test_labels'][0]) > 1:
        for i in range(len(ans[0])):
            name.append("read_%d"%i)
    else:
        name.append('real')
    for i in range(len(ans[0])):
        name.append("pre_%d"%i)
    csv = pd.DataFrame(columns=name, data=concatenate((data['test_labels'], ans), axis=1))
    csv.to_csv("./hw/myoutput/" + data_file + "/ResNet%d_"%net_type + data_file + ".csv")

def result_show(history, data_file, net_type):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig("./hw/myoutput/" + data_file + "/ResNet%d_"%net_type + data_file + "_acc.png")

    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig("./hw/myoutput/" + data_file + "/ResNet%d_"%net_type + data_file + "_loss.png")

    # plt.show()
