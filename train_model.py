from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.layers import Dropout, Lambda
from tensorflow.keras.layers import MaxPooling2D, Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import vgg16
import cv2, logging, datetime
import numpy as np
import pandas as pd
import helper as utl
    
# preparing train and validation sets
def prepare_data(data_dir, TRAIN_IMAGES, TEST_IMAGES):
    
    train_images = list(map(lambda x: str(x)+'.jpg', 
                             np.arange(TRAIN_IMAGES)))

    # steering_angles
    df = pd.read_csv('data.txt', delimiter=' |,', engine='python', 
                     header=None, names=['img', 'angle', 'date', 'time'])
    df.drop(['date', 'time'], axis=1, inplace=True)
    df.set_index('img', inplace=True)
    
    df = df.loc[train_images, 'angle']
    
    train = df.iloc[:int(len(train_images)*.85)]
    val = df.iloc[int(len(train_images)*.85):]
    
    return (train.index, val.index), (np.deg2rad(train.values), np.deg2rad(val.values))

# Conv net architecture
def custom_model(dropout_rate=0.3):
    base_model = vgg16.VGG16(include_top=False, weights='imagenet',
                             input_shape=utl.INPUT_SHAPE)
    
    for layer in base_model.layers[:7]:
        layer.trainable = False      
        
    x = Conv2D(512, 7, activation='elu', strides=(2, 2))(base_model.layers[6].output)
    x = Conv2D(256, 5, activation='elu', strides=(1, 1))(x)
    x = Dropout(dropout_rate)(x)
    x = Flatten()(x)
    x = Dense(256, activation='elu')(x)
    x = Dense(128, activation='elu')(x)
    x = Dense(64, activation='elu')(x)
    output_layer = Dense(1, activation='linear')(x)   
    model = Model(inputs=base_model.input, outputs=output_layer)
    return model

# training the model
def train_model(model, train_images, train_angles, val_images, val_angles, 
				batch_size, nb_epochs, lr_rate=1e-04):

    # Callbacks
    log = CSVLogger('log.txt', append=True, separator=';')
    checkpoint = ModelCheckpoint('model-best-{epoch:02d}.h5',
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='auto')
    # Data Generators
    train_angles_dict = dict(zip(train_images, train_angles))
    train_gen = utl.DataGenerator(train_images, train_angles_dict, utl.data_dir, dim=(66, 200), 
    							  batch_size=batch_size, shuffle=True, training=True)
    
    val_angles_dict = dict(zip(val_images, val_angles))
    val_gen = utl.DataGenerator(val_images, val_angles_dict, utl.data_dir, dim=(66, 200), 
    							  batch_size=batch_size, shuffle=False, testing=True)

    model.compile(loss='mean_squared_error', optimizer=Adam(lr=lr_rate))
    model.fit_generator(train_gen,
                        steps_per_epoch=len(train_images)//batch_size,
                        epochs=nb_epochs, 
                        max_queue_size=40,
                        validation_data=val_gen,
                        validation_steps=len(val_images)//batch_size,
                        callbacks=[checkpoint, log],
                        verbose=1,
                        shuffle=True,
                        use_multiprocessing=True,
                        workers=10)
    

if __name__ == '__main__':
    (train_images, val_images), (train_angles, val_angles) = prepare_data(utl.data_dir, utl.TRAIN_IMAGES, utl.TEST_IMAGES)


    # initializing the custom model
    model = custom_model()
    #train_model(model, train_images, train_angles, val_images, val_angles, batch_size=32, nb_epochs=15)
    print(model.summary())


