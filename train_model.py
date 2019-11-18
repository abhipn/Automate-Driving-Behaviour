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
    
def load_filenames(train_n, dev_n, test_n):
    """
    Loading file names for train, cv and test sets.
    """
    dev_n  = train_n + dev_n + 1
    test_n = dev_n + test_n + 1
    
    train_imgs = list(map(lambda x: str(x) + '.jpg', np.arange(train_n)))
    dev_imgs = list(map(lambda x: str(x) + '.jpg', np.arange(train_n, dev_n)))
    test_imgs = list(map(lambda x: str(x) + '.jpg', np.arange(dev_n, test_n)))
    return train_imgs, dev_imgs, test_imgs

def load_steering_angles(train_imgs, dev_imgs, test_imgs):
    """
    Loading angles for train, cv and test set.
    """
    columns = ['imgs', 'angles', 'date', 'time']
    
    data_df = pd.read_csv('data.txt', sep=' |,', engine='python', 
                          header=None, names=columns).set_index('imgs')
    
    train_angles = data_df.loc[train_imgs]['angles'].to_numpy()
    dev_angles = data_df.loc[dev_imgs]['angles'].to_numpy()
    test_angles = data_df.loc[test_imgs]['angles'].to_numpy()
    
    return train_angles, dev_angles, test_angles



def custom_model(dropout_rate=0.3):
    """
    An Hybrid Model.
    """
    base_model = vgg16.VGG16(include_top=False, weights='imagenet',
                             input_shape=utl.INPUT_SHAPE)
    
    for layer in base_model.layers[:7]:
        layer.trainable = False      
        
    x = Conv2D(256, 5, activation='elu', strides=(2, 2))(base_model.layers[6].output)
    x = Conv2D(128, 3, activation='elu', strides=(1, 1))(x)
    x = Dropout(dropout_rate)(x)
    x = Flatten()(x)
    x = Dense(256, activation='elu')(x)
    x = Dense(128, activation='elu')(x)
    x = Dense(64, activation='elu')(x)
    output_layer = Dense(1, activation='linear')(x)   
    model = Model(inputs=base_model.input, outputs=output_layer)
    return model

def train_model(model, train_images, train_angles, 
		val_images, 
                val_angles, 
		batch_size, 
                nb_epochs, 
                lr_rate=1e-04):
    """
    Training the model.
    """

    # Callbacks
    log = CSVLogger('log.txt', append=True, separator=';')
    checkpoint = ModelCheckpoint('model-best-{epoch:02d}.h5',
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='auto')
    # Data Generators
    train_angles_dict = dict(zip(train_imgs, train_angles))
    dev_angles_dict = dict(zip(dev_imgs, dev_angles))
    
    train_gen = utl.DataGenerator(train_imgs, train_angles_dict, utl.data_dir, 
                              dim=(utl.IMAGE_HEIGHT, utl.IMAGE_WIDTH), 
                              batch_size=64, 
                              shuffle=True, 
                              training=True)
    dev_gen = utl.DataGenerator(dev_imgs, dev_angles_dict, utl.data_dir, 
                            dim=(utl.IMAGE_HEIGHT, utl.IMAGE_WIDTH), 
                            batch_size=64, 
                            shuffle=False,
                            training=False)

    # Compiling and training the model.
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=lr_rate))    
    model.fit_generator(train_gen,
                        steps_per_epoch=utl.TRAIN_IMAGES//batch_size,
                        epochs=nb_epochs, 
                        max_queue_size=30,
                        validation_data=dev_gen,
                        validation_steps=utl.TEST_IMAGES//batch_size,
                        callbacks=[checkpoint, log],
                        verbose=1,
                        shuffle=True,
                        use_multiprocessing=True,
                        workers=10)


if __name__ == '__main__':
    train_imgs, dev_imgs, test_imgs = load_filenames(utl.TRAIN_IMAGES, utl.DEV_IMAGES, utl.TEST_IMAGES)
    train_angles, dev_angles, test_angles = load_steering_angles(train_imgs, dev_imgs, test_imgs)
    
    pickle.dump(test_imgs, open('test_imgs.pkl', 'wb'))
    pickle.dump(test_angles, open('test_angles.pkl', 'wb'))

    # initializing the custom model
    model = custom_model()
    train_model(model, train_imgs, train_angles, dev_imgs, dev_angles, batch_size=64, nb_epochs=15)


