from keras.layers import Conv2D, BatchNormalization, Activation, Input, Dense, AveragePooling2D, Flatten, MaxPooling2D, Add
from keras import Model

# import os
# # 防止输出提示
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# # 使用 GPU 0，1
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

class Resnet:
    def resnet_18(self, W, classes, task_type):
        ## B H W C
        ## Image 224 224 1
        inputs = Input(shape=(224, 224, W))

        ## red 112 112 64
        x = Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding="same")(inputs)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)

        ## white 56 56 64
        x = MaxPooling2D(pool_size=(3, 3), strides=2, padding="same")(x)

        y = x

        ## blue 56 56 64
        x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)

        x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Add()([x, y])
        x = Activation("relu")(x)

        y = x

        x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)

        x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Add()([x, y])
        x = Activation("relu")(x)

        y = x
        y = Conv2D(filters=128, kernel_size=(1, 1), strides=2, padding="same")(y)
        y = BatchNormalization(axis=-1)(y)
        y = Activation("relu")(y)

        ## orange 28 28 128
        x = Conv2D(filters=128, kernel_size=(3, 3), strides=2, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)

        x = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Add()([x, y])
        x = Activation("relu")(x)

        y = x

        x = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)

        x = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Add()([x, y])
        x = Activation("relu")(x)

        y = x
        y = Conv2D(filters=256, kernel_size=(1, 1), strides=2, padding="same")(y)
        y = BatchNormalization(axis=-1)(y)
        y = Activation("relu")(y)

        ## yellow 14 14 256
        x = Conv2D(filters=256, kernel_size=(3, 3), strides=2, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)

        x = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Add()([x, y])
        x = Activation("relu")(x)

        y = x

        x = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)

        x = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Add()([x, y])
        x = Activation("relu")(x)

        y = x
        y = Conv2D(filters=512, kernel_size=(1, 1), strides=2, padding="same")(y)
        y = BatchNormalization(axis=-1)(y)
        y = Activation("relu")(y)

        ## pink 7 7 512
        x = Conv2D(filters=512, kernel_size=(3, 3), strides=2, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)

        x = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Add()([x, y])
        x = Activation("relu")(x)

        y = x

        x = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)

        x = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Add()([x, y])
        x = Activation("relu")(x)

        ## grey 1 1 512
        x = AveragePooling2D(pool_size=(7, 7))(x)

        x = Flatten()(x)  # 512
        x = Dense(1000, activation='relu')(x)  # 512 1000
        if (len(task_type) > 1):
            predictions = Dense(classes, activation='sigmoid')(x)  # 1000 classes
        else:
            predictions = Dense(classes, activation='softmax')(x)  # 1000 classes

        model = Model(inputs=inputs, outputs=predictions)
        return model

    def resnet_50(self, W, classes, task_type):
        ## B H W C
        ## Image 224 224 1
        inputs = Input(shape=(224, 224, W))

        ## red 112 112 64
        x = Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding="same")(inputs)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)

        ## white 56 56 64
        x = MaxPooling2D(pool_size=(3, 3), strides=2, padding="same")(x)

        y = x
        y = Conv2D(filters=256, kernel_size=(1, 1), strides=1, padding="same")(y)
        y = BatchNormalization(axis=-1)(y)
        y = Activation("relu")(y)

        ## blue 56 56 64
        x = Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)

        x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)

        x = Conv2D(filters=256, kernel_size=(1, 1), strides=1, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Add()([x, y])
        x = Activation("relu")(x)

        y = x

        x = Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)

        x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)

        x = Conv2D(filters=256, kernel_size=(1, 1), strides=1, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Add()([x, y])
        x = Activation("relu")(x)

        y = x

        x = Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)

        x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)

        x = Conv2D(filters=256, kernel_size=(1, 1), strides=1, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Add()([x, y])
        x = Activation("relu")(x)

        y = x
        y = Conv2D(filters=512, kernel_size=(1, 1), strides=2, padding="same")(y)
        y = BatchNormalization(axis=-1)(y)
        y = Activation("relu")(y)

        ## orange 28 28 128
        ## 1
        x = Conv2D(filters=128, kernel_size=(1, 1), strides=1, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)

        x = Conv2D(filters=128, kernel_size=(3, 3), strides=2, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)

        x = Conv2D(filters=512, kernel_size=(1, 1), strides=1, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Add()([x, y])
        x = Activation("relu")(x)

        y = x

        ## 2
        x = Conv2D(filters=128, kernel_size=(1, 1), strides=1, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)

        x = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)

        x = Conv2D(filters=512, kernel_size=(1, 1), strides=1, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Add()([x, y])
        x = Activation("relu")(x)

        y = x

        ## 3
        x = Conv2D(filters=128, kernel_size=(1, 1), strides=1, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)

        x = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)

        x = Conv2D(filters=512, kernel_size=(1, 1), strides=1, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Add()([x, y])
        x = Activation("relu")(x)

        y = x

        ## 4
        x = Conv2D(filters=128, kernel_size=(1, 1), strides=1, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)

        x = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)

        x = Conv2D(filters=512, kernel_size=(1, 1), strides=1, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Add()([x, y])
        x = Activation("relu")(x)

        y = x
        y = Conv2D(filters=1024, kernel_size=(1, 1), strides=2, padding="same")(y)
        y = BatchNormalization(axis=-1)(y)
        y = Activation("relu")(y)

        ## yellow 14 14 256
        ## 1
        x = Conv2D(filters=256, kernel_size=(1, 1), strides=1, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)

        x = Conv2D(filters=256, kernel_size=(3, 3), strides=2, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)

        x = Conv2D(filters=1024, kernel_size=(1, 1), strides=1, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Add()([x, y])
        x = Activation("relu")(x)

        y = x

        ## 2
        x = Conv2D(filters=256, kernel_size=(1, 1), strides=1, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)

        x = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)

        x = Conv2D(filters=1024, kernel_size=(1, 1), strides=1, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Add()([x, y])
        x = Activation("relu")(x)

        y = x

        ## 3
        x = Conv2D(filters=256, kernel_size=(1, 1), strides=1, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)

        x = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)

        x = Conv2D(filters=1024, kernel_size=(1, 1), strides=1, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Add()([x, y])
        x = Activation("relu")(x)

        y = x

        ## 4
        x = Conv2D(filters=256, kernel_size=(1, 1), strides=1, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)

        x = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)

        x = Conv2D(filters=1024, kernel_size=(1, 1), strides=1, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Add()([x, y])
        x = Activation("relu")(x)

        y = x

        ## 5
        x = Conv2D(filters=256, kernel_size=(1, 1), strides=1, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)

        x = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)

        x = Conv2D(filters=1024, kernel_size=(1, 1), strides=1, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Add()([x, y])
        x = Activation("relu")(x)

        y = x

        ## 6
        x = Conv2D(filters=256, kernel_size=(1, 1), strides=1, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)

        x = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)

        x = Conv2D(filters=1024, kernel_size=(1, 1), strides=1, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Add()([x, y])
        x = Activation("relu")(x)

        y = x
        y = Conv2D(filters=2048, kernel_size=(1, 1), strides=2, padding="same")(y)
        y = BatchNormalization(axis=-1)(y)
        y = Activation("relu")(y)

        ## pink 7 7 512
        ##1
        x = Conv2D(filters=512, kernel_size=(1, 1), strides=1, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)

        x = Conv2D(filters=512, kernel_size=(3, 3), strides=2, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)

        x = Conv2D(filters=2048, kernel_size=(1, 1), strides=1, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Add()([x, y])
        x = Activation("relu")(x)

        y = x

        ##2
        x = Conv2D(filters=512, kernel_size=(1, 1), strides=1, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)

        x = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)

        x = Conv2D(filters=2048, kernel_size=(1, 1), strides=1, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Add()([x, y])
        x = Activation("relu")(x)

        y = x

        ##3
        x = Conv2D(filters=512, kernel_size=(1, 1), strides=1, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)

        x = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)

        x = Conv2D(filters=2048, kernel_size=(1, 1), strides=1, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Add()([x, y])
        x = Activation("relu")(x)

        ## grey 1 1 2048
        x = AveragePooling2D(pool_size=(7, 7))(x)

        x = Flatten()(x)  # 2048
        x = Dense(1000, activation='relu')(x)  # 2048 1000
        if (len(task_type) > 1):
            predictions = Dense(classes, activation='sigmoid')(x)  # 1000 classes
        else:
            predictions = Dense(classes, activation='softmax')(x)  # 1000 classes

        model = Model(inputs=inputs, outputs=predictions)
        return model
