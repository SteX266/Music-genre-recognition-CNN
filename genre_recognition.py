import numpy as np
import os
import matplotlib.pyplot as plt
import librosa
from keras.layers import (Input,  Dense, Activation, BatchNormalization, Flatten,
                          Conv2D, MaxPooling2D)
from keras.models import Model
from keras.optimizers import Adam
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from pydub import AudioSegment
import shutil
from keras.preprocessing.image import ImageDataGenerator
import random

from tensorflow.python.ops.init_ops_v2 import glorot_uniform


def GenreModel(input_shape=(288, 432, 4), classes=9):
    X_input = Input(input_shape)

    X = Conv2D(8, kernel_size=(3, 3), strides=(1, 1))(X_input)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2))(X)

    X = Conv2D(16, kernel_size=(3, 3), strides=(1, 1))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2))(X)

    X = Conv2D(32, kernel_size=(3, 3), strides=(1, 1))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2))(X)

    X = Conv2D(64, kernel_size=(3, 3), strides=(1, 1))(X)
    X = BatchNormalization(axis=-1)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2))(X)

    X = Conv2D(128, kernel_size=(3, 3), strides=(1, 1))(X)
    X = BatchNormalization(axis=-1)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2))(X)


    X = Flatten()(X)

    #X = Dropout(rate=0.3)

    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=9))(X)
    model = Model(inputs=X_input, outputs=X, name='GenreModel')

    return model


def divide_train_data(genres):
    directory = "/content/spectrograms_3sec/train/"
    for g in genres:
        filenames = os.listdir(os.path.join(directory, f"{g}"))
        random.shuffle(filenames)
        test_files = filenames[0:180]

        for f in test_files:
            shutil.move(directory + f"{g}" + "/" + f, "/content/spectrograms_3sec/test/" + f"{g}")

def split_files(genres):
    i=0
    for g in genres:
        j = 0
        print(f"{g}")
        for filename in os.listdir(os.path.join('/content/Data/genres_original', f"{g}")):

            song = os.path.join(f'/content/Data/genres_original/{g}', f'{filename}')
            j = j + 1
            for w in range(0, 10):
                i = i + 1
                # print(i)
                t1 = 3 * (w) * 1000
                t2 = 3 * (w + 1) * 1000
                newAudio = AudioSegment.from_wav(song)
                new = newAudio[t1:t2]
                new.export(f'/content/genres_original_3sec/{g}/{g + str(j) + str(w)}.wav', format="wav")


def make_spectrogram(genres):
    for g in genres:
        j = 0
        print(g)
        for filename in os.listdir(os.path.join('/content/genres_original_3sec', f"{g}")):
            song = os.path.join(f'/content/genres_original_3sec/{g}', f'{filename}')
            j = j + 1
            if(j < 0):
                continue
            y, sr = librosa.load(song, duration=3)
            # print(sr)
            mels = librosa.feature.melspectrogram(y=y, sr=sr)
            fig = plt.Figure()
            canvas = FigureCanvas(fig)
            p = plt.imshow(librosa.power_to_db(mels, ref=np.max))
            plt.savefig(f'/content/spectrograms_3sec/{g}/{g + str(j)}.png')



if __name__ == '__main__':
    genres = '' \
             'country disco pop hiphop metal reggae rock'
    genres = genres.split()
  #  split_files(genres)
 #   make_spectrogram(["rock"])
   # divide_train_data(genres)


    train_dir = "/content/spectrograms_3sec/train/"
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator = train_datagen.flow_from_directory(train_dir, target_size=(288, 432), color_mode="rgba",
                                                        class_mode='categorical', batch_size=128)

    validation_dir = "/content/spectrograms_3sec/test/"
    vali_datagen = ImageDataGenerator(rescale=1. / 255)
    vali_generator = vali_datagen.flow_from_directory(validation_dir, target_size=(288, 432), color_mode='rgba',
                                                      class_mode='categorical', batch_size=128)

    model = GenreModel(input_shape=(288, 432, 4), classes=9)
    opt = Adam(learning_rate=0.0005)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit_generator(train_generator, epochs=70, validation_data=vali_generator)


