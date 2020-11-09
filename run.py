from keras.models import Sequential
from keras.layers import Cropping2D, Conv2D, Flatten, Dense, Dropout, BatchNormalization, LeakyReLU, MaxPool2D
from keras.optimizers import Adam
from keras import regularizers
import tensorflow as tf
from keras.layers import Lambda
import cv2
import os
import numpy as np
import scipy
from keras.preprocessing import image
L2NormConst = 0.001
model = Sequential()

model = Sequential()


model.add(Cropping2D(((80, 0), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: tf.image.resize(x, (66, 200))))

model.add(Conv2D(24, (5, 5), activation='relu', input_shape=(
    200, 200, 3), strides=(2, 2), kernel_regularizer=regularizers.l2(0.001)))

model.add(BatchNormalization())

model.add(Conv2D(36, (5, 5), activation='relu', strides=(
    2, 2), kernel_regularizer=regularizers.l2(0.001)))
model.add(BatchNormalization())

model.add(Conv2D(48, (5, 5), activation='relu', strides=(
    2, 2), kernel_regularizer=regularizers.l2(0.001)))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), activation='relu',
                 kernel_regularizer=regularizers.l2(0.001)))
model.add(BatchNormalization())

model.add(Conv2D(128, (3, 3), activation='relu',
                 kernel_regularizer=regularizers.l2(0.001)))
model.add(BatchNormalization())

model.add(Flatten())

model.add(Dense(1164, activation='relu',
                kernel_regularizer=regularizers.l2(0.001)))

model.add(Dense(200, activation='relu',
                kernel_regularizer=regularizers.l2(0.001)))

model.add(Dense(128, activation='relu',
                kernel_regularizer=regularizers.l2(0.001)))

model.add(Dense(50, activation='relu', kernel_regularizer=regularizers.l2(0.001)))


model.add(Dense(1, activation='tanh', kernel_regularizer=regularizers.l2(0.001)))

model.load_weights("/Users/kushagra/Downloads/weightsfinal.h5")
img_str = cv2.imread("/Users/kushagra/Downloads/steering_wheel_image.jpg", 0)
rows, cols = img_str.shape
smoothed_angle = 0
i = 43000
while(cv2.waitKey(1) != ord('q') and i < 44000):
    file = os.path.exists(
        "/Users/kushagra/Downloads/driving_dataset/" + str(i) + ".jpg")
    if(file):
        img1 = image.load_img(
            "/Users/kushagra/Downloads/driving_dataset/" + str(i) + ".jpg", color_mode='rgb')
        img1 = image.img_to_array(img1)/255.0
        img = image.load_img("/Users/kushagra/Downloads/driving_dataset/" +
                             str(i) + ".jpg", color_mode='rgb', target_size=[160, 320])
        img = image.img_to_array(img)/255.0
        img_resh = np.reshape(img, [1, 160, 320, 3])
        degrees = float(model.predict(img_resh) * 180.0 / scipy.pi)
        print("Predicted steering angle: " + str(degrees) + " degrees")
        cv2.imshow("frame", cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))
        smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (
            degrees - smoothed_angle) / abs(degrees - smoothed_angle)
#rotated = imutils.rotate_bound(img_str, degrees)
        M = cv2.getRotationMatrix2D((cols/2, rows/2), -smoothed_angle, 1)
        dst = cv2.warpAffine(img_str, M, (cols, rows))
        cv2.imshow("steering wheel", dst)
    i += 1
cv2.destroyAllWindows()
