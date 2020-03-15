from keras import Input, Model
from keras import backend as K
from keras.layers import Conv2D, Flatten, Dense, MaxPool2D


class ClassificationNet:

    @staticmethod
    def build(width_height_channel, num_classes):

        (width, height, channel) = width_height_channel
        if K.image_data_format() == 'channels_first':
            input_shape = (channel, height, width)
        else:
            input_shape = (height, width, channel)

        inputs = Input(shape=input_shape)

        x = Conv2D(32, kernel_size=(5, 5), padding='same', activation='relu')(inputs)
        x = MaxPool2D((2, 2), strides=(2, 2), padding='same')(x)
        x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
        x = MaxPool2D((2, 2), strides=(2, 2), padding='same')(x)
        x = Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
        x = MaxPool2D((2, 2), strides=(2, 2), padding='same')(x)
        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(num_classes, activation='sigmoid', name='classification_net_output')(x)

        model = Model(inputs=inputs, outputs=x, name='ClassificationNet')

        return model
