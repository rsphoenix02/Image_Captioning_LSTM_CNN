from vgg16 import VGG16
from keras.applications import inception_v3
import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Input, LSTM, Embedding, TimeDistributed, Dense, RepeatVector, concatenate, Activation, Flatten
from keras.utils import pad_sequences
from keras.utils import load_img, img_to_array
from keras.callbacks import ModelCheckpoint
import pickle as pickle
import os



EMBEDDING_DIM = 128


class CaptionGenerator():

    def __init__(self):
        self.max_cap_len = None
        self.vocab_size = None
        self.index_word = None
        self.word_index = None
        self.total_samples = None
        self.encoded_images = pickle.load(open(os.path.join(os.path.split(os.path.dirname(__file__))[0], "encoded_images.p"), 'rb'))
        self.variable_initializer()

    def variable_initializer(self):
        df = pd.read_csv(os.path.join(os.path.split(os.path.dirname(__file__))[0], "Flickr8k_text", "flickr_8k_train_dataset"), delimiter='\t')
        nb_samples = df.shape[0]
        iter = df.iterrows()
        # for i in iter:
        #     print(i)
        # print(type(iter))
        caps = []
        for i in range(nb_samples):
            x = iter.__next__()
            caps.append(x[1][1])

        self.total_samples=0
        for text in caps:
            self.total_samples+=len(text.split())-1
        print("Total samples : "+str(self.total_samples))
        
        words = [txt.split() for txt in caps]
        unique = []
        for word in words:
            unique.extend(word)

        unique = list(set(unique))
        self.vocab_size = len(unique)
        self.word_index = {}
        self.index_word = {}
        for i, word in enumerate(unique):
            self.word_index[word]=i
            self.index_word[i]=word

        max_len = 0
        for caption in caps:
            if(len(caption.split()) > max_len):
                max_len = len(caption.split())
        self.max_cap_len = max_len
        print("Vocabulary size: "+str(self.vocab_size))
        print("Maximum caption length: "+str(self.max_cap_len))
        print("Variables initialization done!")


    def data_generator(self, batch_size = 32):
        partial_caps = []
        next_words = []
        images = []
        print("Generating data...")
        gen_count = 0
        df = pd.read_csv(os.path.join(os.path.split(os.path.dirname(__file__))[0], "Flickr8k_text", "flickr_8k_train_dataset"), delimiter='\t')
        nb_samples = df.shape[0]
        iter = df.iterrows()
        caps = []
        imgs = []
        for i in range(nb_samples):
            x = iter.__next__()
            caps.append(x[1][1])
            imgs.append(x[1][0])

        print(df.head())


        total_count = 0
        while 1:
            image_counter = -1
            for text in caps:
                image_counter+=1
                current_image = self.encoded_images[imgs[image_counter]]
                for i in range(len(text.split())-1):
                    total_count+=1
                    partial = [self.word_index[txt] for txt in text.split()[:i+1]]
                    partial_caps.append(partial)
                    next = np.zeros(self.vocab_size)
                    next[self.word_index[text.split()[i+1]]] = 1
                    next_words.append(next)
                    images.append(current_image)

                    if total_count>=batch_size:
                        next_words = np.asarray(next_words)
                        images = np.asarray(images)
                        partial_caps = pad_sequences(partial_caps, maxlen=self.max_cap_len, padding='post')
                        total_count = 0
                        gen_count+=1
                        print("yielding count: "+str(gen_count))
                        yield [[images, partial_caps], next_words]
                        partial_caps = []
                        next_words = []
                        images = []
        
    def load_image(self, path):
        img = load_img(path, target_size=(224,224))
        x = img_to_array(img)
        return np.asarray(x)


    def create_model(self, ret_model = False):
        # #base_model = VGG16(weights='imagenet', include_top=False, input_shape = (224, 224, 3)) #
        # #base_model.trainable=False #
        # image_model = Sequential()
        # #image_model.add(base_model) #
        # #image_model.add(Flatten()) #
        # image_model.add(Dense(EMBEDDING_DIM, input_dim = 4096, activation='relu', name='dense_1'))

        # image_model.add(RepeatVector(self.max_cap_len))

        # print("Summary of image model: ")
        # print(image_model.summary())
        # for layer in image_model.layers:
        #     print(layer.output_shape)

        image_model_input = Input(shape=(1000,), name='dense_1_input')
        dense_1 = Dense(EMBEDDING_DIM, input_dim = 4096, activation='relu', name='dense_')(image_model_input)
        repeat_vector_1 = RepeatVector(self.max_cap_len)(dense_1)


        # lang_model = Sequential()
        # lang_model.add(Embedding(self.vocab_size, 256, input_length=self.max_cap_len))
        # lang_model.add(LSTM(256,return_sequences=True))
        # lang_model.add(TimeDistributed(Dense(EMBEDDING_DIM)))

        # print("Summary of lang model: ")
        # print(lang_model.summary())
        # for layer in lang_model.layers:
        #     print(layer.output_shape)

        embedding_1_input = Input(shape=(40,), name='embedding_1_input')
        embedding_1 = Embedding(self.vocab_size, 256, input_length=self.max_cap_len)(embedding_1_input)
        lstm_1 = LSTM(256,return_sequences=True)(embedding_1)
        time_distributed_1 = TimeDistributed(Dense(EMBEDDING_DIM))(lstm_1)

        # model = Sequential()
        # # model.add(concatenate([lang_model.output, image_model.output]))
        # model.add(concatenate([image_model.output, lang_model.output]))
        # model.add(LSTM(1000,return_sequences=False))
        # model.add(Dense(self.vocab_size))
        # model.add(Activation('softmax'))

        concat = concatenate([time_distributed_1, repeat_vector_1], name='Concatenate')
        lstm_2 = LSTM(1000,return_sequences=False)(concat)
        dense_3 = Dense(self.vocab_size)(lstm_2)
        activation_1 = Activation('softmax')(dense_3)
        model = Model(inputs=[image_model_input, embedding_1_input], outputs=activation_1,
                    name='Final_output')

        print("Model created!")

        # print("Summary of concatenated model: ")
        # print(model.summary())

        if(ret_model==True):
            return model

        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        return model

    def get_word(self,index):
        return self.index_word[index]
