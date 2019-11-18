# libraries
import numpy as np
import os, cv2
from skimage.io import imread
from tensorflow.keras.utils import Sequence


# Default Configs
data_dir = 'data'

n_images = len(np.sort(os.listdir(data_dir)))

TRAIN_IMAGES = int(n_images*0.85)
DEV_IMAGES = int(n_images*0.075)
TEST_IMAGES = int(n_images*0.075)

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)



## image preprocessing steps

def load_image(data_dir, image_file):
    return imread(os.path.join(data_dir, image_file))

def crop(image):
    """
    Crop the image (removing the sky at the top and the car front at the bottom)
    """
    return image[105:-5, :, :]

def resize(image):
    """
    Resize the image to the input shape used by the network model
    """
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)

def preprocess_image(image):
    image = crop(image)
    image = resize(image)
    return image


## Data Augumentation
def random_brightness(image):
    """
    Randomly adjust brightness of the image
    """
    coeff = 2* np.random.uniform(0,1) 
    image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS) 
    image_HLS = np.array(image_HLS, dtype = np.float64) 
    image_HLS[:,:,1] = image_HLS[:,:,1]*coeff 
    if(coeff>1):
        image_HLS[:,:,1][image_HLS[:,:,1]>255]  = 255 
    else:
        image_HLS[:,:,1][image_HLS[:,:,1]<0]=0
    image_HLS = np.array(image_HLS, dtype = np.uint8)
    image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB)
    return image_RGB


def random_flip(image, steering_angle):
    """
    Randomly flip the image left <-> right, and adjust the steering angle.
    """
    image = cv2.flip(image, 1)
    steering_angle = -steering_angle
    return image, steering_angle

def random_translate(image, steering_angle):
    """
    Randomly shift the image virtially and horizontally (translation).
    """
    range_x, range_y = np.random.randint(10, 100), np.random.randint(10, 60)
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle

    
def random_shadow(image, no_of_shadows=1):
    """
    Randomly add polygon shaped shadows to the image
    """
    image_HLS = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    mask = np.zeros_like(image)     
    imshape = image.shape    
    vertices_list= generate_shadow_coordinates(imshape, no_of_shadows)  
    for vertices in vertices_list:         
        cv2.fillPoly(mask, vertices, 255)    
        image_HLS[:,:,1][mask[:,:,0]==255] = image_HLS[:,:,1][mask[:,:,0]==255]*0.5     
        image_RGB = cv2.cvtColor(image_HLS, cv2.COLOR_HLS2RGB)
    return image_RGB

def generate_shadow_coordinates(imshape, no_of_shadows=1):
    """
    Helper function for above add_shadow main function
    """
    vertices_list=[]    
    for index in range(no_of_shadows):        
        vertex=[]        
        for dimensions in range(np.random.randint(3,15)): 
            vertex.append(( imshape[1]*np.random.uniform(),imshape[0]//3+imshape[0]*np.random.uniform()))
        vertices = np.array([vertex], dtype=np.int32)      
        vertices_list.append(vertices)    
        return vertices_list
    
def augument(image, steering_angle):
    """
    Generate an random augumented image
    """
    random_value = np.random.randint(0, 4)
    if random_value == 0:
        image = random_brightness(image)
    if random_value == 1:
        image, steering_angle = random_flip(image, steering_angle)
    if random_value == 2:
        image, steering_angle = random_translate(image, steering_angle)
    if random_value == 3:
        image = random_shadow(image)
    return image, steering_angle


# Data Generator
class DataGenerator(Sequence):
    """
    Data Generator with multiprocessing support.
    """
    def __init__(self, images, angles, data_dir, dim=(66, 200), batch_size=32, n_channels=3,
                 n_classes=1, shuffle=True, training=False):

        self.dim = dim
        self.training = training
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.angles = angles
        self.images = images
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        
    def __len__(self): 
        return len(self.images) // self.batch_size
    
    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        data_IDs = [self.images[idx] for idx in indexes]
        X, y = self.__data_generation(data_IDs)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.images))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def __data_generation(self, data_IDs):
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=float)
        
        for i, ID in enumerate(data_IDs):
            img_file = preprocess_image(load_image(self.data_dir, ID))
            
            if self.training:
                if np.random.randn() < 0.3:
                    X[i] = img_file / 255.
                    y[i] = self.angles[ID]
                else:
                    aug_image, aug_angle = augument(img_file, self.angles[ID])
                    X[i] = aug_image / 255.
                    y[i] = aug_angle
            else:
                X[i] = img_file / 255.
                y[i] = self.angles[ID]
        return X, y
