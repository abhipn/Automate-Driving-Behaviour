from tensorflow.keras.models import load_model
import utils as utl
import train_model as tm
import numpy as np, pandas as pd, os, cv2
from sklearn.metrics import mean_squared_error
from skimage import io
import pickle

def prepare_test():
    """
    Loading test images and steering angles from pickle files.
    """
    test_imgs = pickle.load(open('test_imgs.pkl', 'rb'))
    test_angles = pickle.load(open('test_angles.pkl', 'rb'))
    return test_imgs, test_angles


if __name__ == '__main__':
    
    test_imgs, test_angles = prepare_test()
    print('Loading saved best model :\n')
   
    model = load_model('model-best-14.h5')
    print('Model loaded.')
    img = io.imread('wheel.jpg')
    rows, cols, channels = img.shape
    smoothed_angle = 0

    i = 0
    while (cv2.waitKey(10) != ord('q')):
        image = utl.preprocess_image(utl.load_image(utl.data_dir, test_imgs[i])) / 255.
        degrees = model.predict(image.reshape(-1, 66, 200, 3))[0][0]
        print('Steering angle: ', '<<<Actual>>>', np.round(test_angles[i], 4), '<<<Pred>>>>', np.round(degrees, 3))    
        cv2.imshow('Image',  cv2.cvtColor(io.imread(os.path.join(utl.data_dir, test_imgs[i])), cv2.COLOR_BGR2RGB))
        smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
        M = cv2.getRotationMatrix2D((cols/2, rows/2), -smoothed_angle, 1)
        dst = cv2.warpAffine(img, M, (cols, rows))
        cv2.imshow("steering wheel", dst)
        i += 1
    else:
    	cv2.destroyAllWindows()
