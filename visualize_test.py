from tensorflow.keras.models import load_model
import helper as utl
import train_model as tm
import numpy as np, pandas as pd, os, cv2
from skimage import io



def prepare_test(data_dir, TRAIN_IMAGES, TEST_IMAGES):    
    test_images = list(map(lambda x: str(x)+'.jpg', 
                           np.arange(TRAIN_IMAGES, 
                           TRAIN_IMAGES+TEST_IMAGES)))
    # steering_angles
    df = pd.read_csv('data.txt', delimiter=' |,', 
                     engine='python', 
                     header=None, 
                     names=['img', 'angle', 'date', 'time'])
    
    df.drop(['date', 'time'], axis=1, inplace=True)
    df.set_index('img', inplace=True)
    
    df = df.loc[test_images, 'angle']
    return df.index.to_numpy(), np.deg2rad(df.to_numpy())



if __name__ == '__main__':
    
    test_images, test_angles = prepare_test(utl.data_dir, utl.TRAIN_IMAGES, utl.TEST_IMAGES)
    print('Loading saved best model :\n')
   
    model = load_model('model-best.h5')

    img = io.imread('wheel.png')
    rows, cols, channels = img.shape
    smoothed_angle = 0

    i = 0
    while (cv2.waitKey(10) != ord('q')):
        image = utl.preprocess_image(utl.load_image(utl.data_dir, test_images[i])) / 255.
        degrees = np.rad2deg(model.predict(image.reshape(-1, 66, 200, 3)))[0][0]
        print('Steering angle: ', '<<<Actual>>>', np.round(np.rad2deg(test_angles[i]),4), '<<<Pred>>>>', np.round(degrees, 3))    
        cv2.imshow('Image',  cv2.cvtColor(io.imread(os.path.join(utl.data_dir, test_images[i])), cv2.COLOR_BGR2RGB))
        smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
        M = cv2.getRotationMatrix2D((cols/2, rows/2), -smoothed_angle, 1)
        dst = cv2.warpAffine(img, M, (cols, rows))
        cv2.imshow("steering wheel", dst)
        i += 1

    cv2.destroyAllWindows()