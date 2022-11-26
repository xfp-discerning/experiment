#!/usr/bin/env python
# coding: utf-8
from keras.models import Model
from keras.layers import Input, Dense, LeakyReLU, Dropout
from keras.optimizers import Adam
from keras.backend import clear_session
from keras.utils import to_categorical
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from tqdm.notebook import tqdm
from numpy.random import seed
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os

mean_acc_list = []
max_acc_list = []
mean_loss_list = []
acc_STD_list = []
loss_STD_list = []

# Data Parameters
num_of_classes = 54
data_shape = (119, 1)
threshold_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

# MLP Parameters
times_to_run = 10  # Number of times to run MLP model
mlp_epochs = 40
valid_split = 0.20

# Random Seeds
selection_seed = 150
seed_multiplier = 1000000

path1 = "./data_last/gan_2500_mean3/output_features.csv"
path2 = "./data_last/gan_2500_mean3/output_labels.csv"
path3 = "./data_last/gan_2500_mean3/X_train.csv"
path4 = "./data_last/gan_2500_mean3/Y_train.csv"
path5 = "./data_last/gan_2500_mean3/X_test.csv"
path6 = "./data_last/gan_2500_mean3/Y_test.csv"

train_gan = pd.read_csv(path1, header=None)
labels_gan = pd.read_csv(path2, header=None)

train_real = pd.read_csv(path3, header=None)
labels_real = pd.read_csv(path4, header=None)

x_test = pd.read_csv(path5, header=None)
y_test = pd.read_csv(path6, header=None)

train_gan = np.asarray(train_gan, dtype=np.float32)
labels_gan = np.asarray(labels_gan, dtype=np.int32)
labels_gan = labels_gan.reshape(len(train_gan), )

train_real = np.asarray(train_real, dtype=np.float32)
labels_real = np.asarray(labels_real, dtype=np.int32)
labels_real = labels_real.reshape(len(train_real), )

x_test = np.asarray(x_test, dtype=np.float32)
y_test = np.asarray(y_test, dtype=np.int32)
y_test = y_test.reshape(len(x_test), )

# ### classification in real data
x_train, y_train = shuffle(train_real, labels_real, random_state=selection_seed)

scaler = StandardScaler().fit(x_train)
X_train_transformed = scaler.transform(x_train)
X_test_transformed = scaler.transform(x_test)
Y_train_encoded = to_categorical(y_train)
Y_test_encoded = to_categorical(y_test)

if __name__ == "__main__":
    all_test_loss = []
    all_test_acc = []
    history = []

    for i in tqdm(range(times_to_run)):
        seed(i * seed_multiplier)  # 操作级
        tf.random.set_seed(i * seed_multiplier)  # 图级
        inp = Input(shape=(data_shape[0],), name='ap_features')
        x = Dense(1024, activation=LeakyReLU(alpha=0))(inp)
        x = Dropout(0.3)(x)
        x = Dense(512, activation=LeakyReLU(alpha=0))(x)
        x = Dropout(0.3)(x)
        x = Dense(256, activation=LeakyReLU(alpha=0))(x)
        x = Dropout(0.3)(x)
        x = Dense(128, activation=LeakyReLU(alpha=0))(x)
        output = Dense(54, activation='softmax')(x)
        model = Model(inp, output)

        model.compile(optimizer=Adam(0.0002, 0.05),
                      # learning rate, the exponential decay rate for the 1st moment estimates
                      # 学习率      , 一阶矩估计的指数衰减率
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        history_temp = model.fit(X_train_transformed,
                                 Y_train_encoded,
                                 epochs=mlp_epochs - 10,
                                 batch_size=32,
                                 validation_split=0.30,
                                 verbose=0)
        history.append(history_temp)
        test_loss, test_acc = model.evaluate(X_test_transformed,
                                             Y_test_encoded,
                                             verbose=0)
        print("#{} Test acc:".format(i), test_acc)

        all_test_acc.append(test_acc)
        all_test_loss.append(test_loss)
        del model
        clear_session()

    from sklearn import neighbors

    knn = neighbors.KNeighborsClassifier(n_neighbors=1)
    knn.fit(train_real, labels_real)
    knn_acc = knn.score(x_test, y_test)
    print('the accurancy is: %.4f' % knn_acc)

    trainacc = []
    trainloss = []
    valacc = []
    valloss = []
    for i in range(len(history)):
        trainacc.append(history[i].history['accuracy'])
        trainloss.append(history[i].history['loss'])
        valacc.append(history[i].history['val_accuracy'])
        valloss.append(history[i].history['val_loss'])

    acc = np.mean(trainacc, axis=0)
    val_acc = np.mean(valacc, axis=0)
    loss = np.mean(trainloss, axis=0)
    val_loss = np.mean(valloss, axis=0)
    epochs = range(1, len(acc) + 1)

    now = datetime.datetime.now()
    os.mkdir('./gan_3/{}_real'.format(now.strftime("%Y%m%d-%H%M%S")))
    plt.figure()
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()  # 作用是加上图例，很有必要
    plt.savefig("./gan_3/{}_real/real_TRandVAL_acc.png".format(now.strftime("%Y%m%d-%H%M%S")))
    plt.figure()  # 创建新图
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig("./gan_3/{}_real/real_TRandVAL_loss.png".format(now.strftime("%Y%m%d-%H%M%S")))

    for threshold in threshold_list:
        ### classification in gen+real
        train_gan1 = train_gan
        train_gan1[110 - abs(train_gan1) < threshold] = -110  # 空值滤波
        x_train_gan1, y_train_gan = shuffle(train_gan1, labels_gan, random_state=selection_seed)

        X_train_gan_transformed = scaler.transform(x_train_gan1)
        Y_train_gan_encoded = to_categorical(y_train_gan)

        all_test_loss_gan = []
        all_test_acc_gan = []
        ganhistory = []

        print("这是第%d个" % (threshold * 10 + 1))
        for i in tqdm(range(times_to_run)):
            seed(i * seed_multiplier)
            tf.random.set_seed(i * seed_multiplier)

            inp = Input(shape=(data_shape[0],), name='ap_features')
            x = Dense(1024, activation=LeakyReLU(alpha=0.2))(inp)
            x = Dropout(0.3)(x)
            x = Dense(512, activation=LeakyReLU(alpha=0.2))(x)
            x = Dropout(0.3)(x)
            x = Dense(256, activation=LeakyReLU(alpha=0.2))(x)
            x = Dropout(0.3)(x)
            x = Dense(128, activation=LeakyReLU(alpha=0.2))(x)
            output = Dense(54, activation='softmax')(x)
            model = Model(inp, output)

            model.compile(optimizer=Adam(0.0002, 0.5),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

            ganhistorytemp = model.fit(X_train_gan_transformed,
                                       Y_train_gan_encoded,
                                       epochs=mlp_epochs - 10,
                                       batch_size=32,
                                       validation_split=0.3,
                                       verbose=0)
            ganhistory.append(ganhistorytemp)

            test_loss, test_acc = model.evaluate(X_test_transformed,
                                                 Y_test_encoded,
                                                 verbose=0)
            print("#{} Test acc:".format(i), test_acc)

            all_test_acc_gan.append(test_acc)
            all_test_loss_gan.append(test_loss)
            del model
            clear_session()

        knn1 = neighbors.KNeighborsClassifier(n_neighbors=1)
        knn1.fit(train_gan, labels_gan)
        acc1 = knn1.score(x_test, y_test)
        print()
        print('the accurancy is: %.4f' % acc1)

        gan_trainacc = []
        gan_trainloss = []
        gan_valacc = []
        gan_valloss = []
        for i in range(len(ganhistory)):
            gan_trainacc.append(ganhistory[i].history['accuracy'])
            gan_trainloss.append(ganhistory[i].history['loss'])
            gan_valacc.append(ganhistory[i].history['val_accuracy'])
            gan_valloss.append(ganhistory[i].history['val_loss'])

        acc = np.mean(gan_trainacc, axis=0)
        val_acc = np.mean(gan_valacc, axis=0)
        loss = np.mean(gan_trainloss, axis=0)
        val_loss = np.mean(gan_valloss, axis=0)
        epochs = range(1, len(acc) + 1)

        os.mkdir('./gan_3/{}_gen_threshold({})'.format(now.strftime("%Y%m%d-%H%M%S"), threshold))
        plt.figure()
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()  # 作用是加上图例，很有必要
        plt.savefig("./gan_3/{}_gen_threshold({})/gen+real_TRandVAL_acc.png".format(now.strftime("%Y%m%d-%H%M%S"),
                                                                                    threshold))
        plt.figure()  # 创建新图
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.savefig("./gan_3/{}_gen_threshold({})/gen+real_TRandVAL_loss.png".format(now.strftime("%Y%m%d-%H%M%S"),
                                                                                     threshold))

        AccMean = np.mean(all_test_acc)
        LossMean = np.mean(all_test_loss)
        AccStd = np.std(all_test_acc)
        LossStd = np.std(all_test_loss)

        GanAccMean = np.mean(all_test_acc_gan)
        GanLossMean = np.mean(all_test_loss_gan)
        GanAccStd = np.std(all_test_acc_gan)
        GanLossStd = np.std(all_test_loss_gan)

        lines = list()
        lines.append("Accuracy mean: {}".format(AccMean))
        lines.append("Loss mean: {}".format(LossMean))
        lines.append("Accuracy STD: {}".format(AccStd))
        lines.append("Loss STD: {} \n".format(LossStd))
        lines.append("Maximum Accuracy: {}".format(np.max(all_test_acc)))
        lines.append("Loss of Maximum Accuracy: {}\n".format(
            all_test_loss[np.argmax(all_test_acc)]))
        lines.append("acc of the KNN: %.4f" % knn_acc)
        # for m in range(len(acc)):
        #     lines.append("K = {}, acc = {}".format(m+1,acc[m]))

        lines.append("\n ================== \n")

        lines.append("Accuracy mean: {}".format(GanAccMean))
        lines.append("Loss mean: {}".format(GanLossMean))
        lines.append("Accuracy STD: {}".format(GanAccStd))
        lines.append("Loss STD: {} \n".format(GanLossStd))
        lines.append("Maximum Accuracy: {}".format(np.max(all_test_acc_gan)))
        lines.append("Loss of Maximum Accuracy: {}\n".format(
            all_test_loss_gan[np.argmax(all_test_acc_gan)]))
        lines.append("acc of the KNN: %.4f" % acc1)
        # for n in range(len(acc1)):
        #     lines.append("K = {}, acc = {}".format(n+1,acc1[n]))

        file_dir = "./gan_3/{}_gen_threshold({})/test.txt".format(now.strftime("%Y%m%d-%H%M%S"), threshold)
        with open(file_dir, "w") as filehandle:
            for items in lines:
                filehandle.write('%s\n' % items)

        mean_acc_list.append(GanAccMean)
        mean_loss_list.append(GanLossMean)
        acc_STD_list.append(GanAccStd)
        loss_STD_list.append(GanLossStd)
        max_acc_list.append(np.max(all_test_acc_gan))

    # 整体分析
    threshold_list = [str(i) for i in threshold_list]  # 将整形转为string，使得横坐标完全显示
    plt.figure()
    plt.plot(threshold_list, mean_acc_list, 'r', label="mean acc")
    plt.plot(threshold_list, max_acc_list, 'b', label="max acc")
    plt.xlabel("threshold")
    plt.ylabel("acc")
    plt.title('mean_acc and max_acc')
    plt.legend()
    plt.savefig("./gan_3/mean_acc and max_acc.png")

    plt.figure()
    plt.plot(threshold_list, acc_STD_list, "r", label="acc STD")
    plt.plot(threshold_list, loss_STD_list, "b", label="loss STD")
    plt.title('acc_STD and loss_STD')
    plt.legend()
    plt.savefig("./gan_3/acc_STD and loss_STD.png")

    plt.figure()
    plt.plot(threshold_list, mean_loss_list, 'r', label="mean loss")
    plt.title('mean_loss')
    plt.legend()
    plt.savefig("./gan_3/mean_loss.png")
