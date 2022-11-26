#!/usr/bin/env python
# coding: utf-8

# In[13]:


from keras.models import Sequential, Model
from keras.layers import Input, Dense, Reshape, Flatten, LeakyReLU
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras.backend import clear_session
from keras.utils import to_categorical
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from numpy.random import seed
import pandas as pd
import numpy as np

# In[14]:


# Data Parameters
num_of_classes = 54
data_shape = (119, 1)

# MLP Parameters
times_to_run = 5  # Number of times to run MLP model
mlp_epochs = 30
valid_split = 0.20

# GAN Parameters
latent_dim = 200
gan_epochs = 2500

# Random Seeds
selection_seed = 150
seed_multiplier = 1000000

# In[15]:

if __name__ == "__main__":
    path = "./UJIIndoorLoc/children_13/sorted/00_sorted.csv"
    train_df = pd.read_csv(path, header=0)
    print(train_df.shape)
    a = train_df.mean()
    print(a["WAP004"])
    for i in train_df.columns[:520]:
        if a[i] == -110:
            del train_df[i]
    print(train_df.shape)
    train_df["REF"] = pd.factorize(train_df["REF"])[0].astype(int)  # 将标签映射到顺序数字上
    labels = train_df.REF.values
    features = train_df.drop(columns=['TIMESTAMP', 'PHONEID', 'USERID', 'RELATIVEPOSITION',
                                      'SPACEID', 'BUILDINGID', 'FLOOR', 'LATITUDE', 'LONGITUDE',
                                      'BF', 'REF']).values
    input_shape = features.shape[1:]

    # In[16]:

    X_tr, X_test, Y_tr, Y_test = train_test_split(features,
                                                  labels,
                                                  test_size=0.2,
                                                  random_state=selection_seed,
                                                  # random_state：可以理解为随机数种子，主要是为了复现结果而设置
                                                  stratify=labels)  # stratify保证测试集中，所有类别的齐全

    # In[17]:

    # 统计各标签的样本数量
    unique, count = np.unique(Y_tr, return_counts=True)
    data_count = dict(zip(unique, count))
    s_mean = int(count.mean()) + 1

    # In[18]:

    X_train = []
    Z_train = []  # This is the same as X_train, but it's used for training the GAN
    Y_train = []

    for idx in range(54):
        number_filter = np.where(Y_tr == idx)
        X_filtered, Y_filtered = X_tr[number_filter], Y_tr[number_filter]

        Z_train.append(X_filtered)  # append添加是将容器看作整体来进行添加，但extend是将容器打碎后添加（加入的只是元素）
        X_train.extend(X_filtered)
        Y_train.extend(Y_filtered)
    # X_train,Z_train
    X_train = np.asarray(X_train, dtype=np.float32)
    Y_train = np.asarray(Y_train, dtype=np.float32)
    # X_train
    # random.shuffle只能对一维list和两维list进行数据打乱
    # numpy.random.shuffle可以对列表和数组进行数据打乱
    # from sklearn.utils import shuffle中的shuffle可以直接打乱
    X_train, Y_train = shuffle(X_train, Y_train)

    # 将类别向量映射为二值类别矩阵, 用于应用到以categorical_crossentropy为目标函数的模型中.
    # to_categorical和pd.get_dummies推荐前者，后者原理更复杂
    Y_train_encoded = to_categorical(Y_train)
    Y_test_encoded = to_categorical(Y_test)
    scaler = StandardScaler()  # from sklearn.preprocessing import StandardScaler
    X_train_transformed = scaler.fit_transform(X_train)
    X_test_transformed = scaler.fit_transform(X_test)


    # In[19]:

    def build_generator():
        model = Sequential()

        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(input_shape), activation='tanh'))
        model.add(Reshape(input_shape))

        #     model.summary()

        noise = Input(shape=(latent_dim,))
        gendata = model(noise)

        return Model(noise, gendata)  # keras.models.Model(input,output) 模型起始输入和最终输出 #Sequential()继承于Model


    def build_discriminator():

        model = Sequential()

        model.add(Flatten(input_shape=data_shape))  # keras.layers.Flatten将数据压成一维的
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))  # 二分类一般使用sigmoid 和softmax

        # model.summary()

        data = Input(shape=data_shape)
        validity = model(data)

        return Model(data, validity)


    def train(epochs, features, batch_size):

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            idx = np.random.randint(0, features.shape[0], batch_size)
            data = features[idx]

            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            gen_data = generator.predict(noise)  # generator

            d_loss_real = discriminator.train_on_batch(data, valid)
            d_loss_fake = discriminator.train_on_batch(gen_data, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)  # 损失函数应该还可以改进

            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            g_loss = combined.train_on_batch(noise, valid)  # combined


    # In[ ]:

    gen_data = []
    gen_label = []

    for i in range(num_of_classes):
        if data_count[i] < s_mean * 2:
            will_to_gen = s_mean * 2 - data_count[i]
            discriminator = build_discriminator()
            discriminator.compile(loss='binary_crossentropy',
                                  optimizer=Adam(0.0002, 0.5),
                                  metrics=['accuracy'])

            generator = build_generator()
            noise = Input(shape=(latent_dim,))
            gendata = generator(noise)
            discriminator.trainable = False  # 禁用了判别器的参数更新
            validity = discriminator(gendata)
            combined = Model(noise, validity)  # 连接生成器和判别器
            combined.compile(loss='binary_crossentropy',
                             optimizer=Adam(0.0002, 0.5))

            scaler = StandardScaler()
            Z_train_transformed = scaler.fit_transform(Z_train[i])
            Z_train_transformed = np.expand_dims(Z_train_transformed, axis=2)  # 扩充维度（*，*，1）

            train(epochs=gan_epochs,
                  features=Z_train_transformed,
                  batch_size=32)

            noise = np.random.normal(0, 1, (will_to_gen, latent_dim))
            gen_data_temp = generator.predict(noise)
            gen_data_temp = np.asarray(gen_data_temp, dtype=np.float32)
            gen_data_temp = np.squeeze(gen_data_temp)  # 删除维度为1 #（1，2，5）==> （2，5）
            gen_data_temp = gen_data_temp.reshape(will_to_gen, data_shape[0])
            gen_data_temp = scaler.inverse_transform(gen_data_temp)  # 将归一化数据转回来

            gen_data.extend(gen_data_temp)
            gen_label_temp = np.tile(i, will_to_gen)
            gen_label.extend(gen_label_temp)

            clear_session()
            # 在每折的开头都需要加上clear_session()。否则上一折的训练集成了这一折的验证集，数据泄露。
            # 同时，不清空的话，那么graph上的node越来越多，内存问题，时间问题都会变得严峻。
            # 可以有效解决模型的内存占用问题
            del discriminator
            del generator
            del combined

    gen_data = np.asarray(gen_data, dtype=np.float32)
    gen_label = np.asarray(gen_label, dtype=np.float32)

    # In[ ]:

    gen_data_reshaped = gen_data.reshape(-1, data_shape[0])
    X_train_gan, Y_train_gan = shuffle(gen_data_reshaped,
                                       gen_label,
                                       random_state=5)
    new_x_train = np.concatenate((X_train, X_train_gan), axis=0)
    y_train = np.concatenate((Y_train, Y_train_gan), axis=0)

    new_x_train, y_train = shuffle(new_x_train, y_train, random_state=15)

    # os.mkdir('./UJIIndoorLoc/gan_2500_mean')
    np.savetxt('./UJIIndoorLoc/gan_2500_mean2/output_features.csv', new_x_train, delimiter=', ', fmt='%f')
    np.savetxt('./UJIIndoorLoc/gan_2500_mean2/output_labels.csv', y_train, delimiter=', ', fmt='%f')
