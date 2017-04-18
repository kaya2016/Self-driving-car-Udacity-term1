# required Packages:
import csv
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np 
import keras
from keras.models import Sequential 
from keras.layers.pooling import MaxPooling2D
from keras.layers import Flatten, Dense ,Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.optimizers import Adam
# the images and the measurments from the data set
lines = []
with open ('./data-udacity/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
sim_images = []
sim_measurments = []
for line in lines:
    for i in range(3):
        source_path = line[i]
        tokens = source_path.split('/')
        filename = tokens[-1]
        local_path = "./data-udacity/IMG/"+filename
        image = cv2.imread(local_path)
        sim_images.append(image)
    # correcction factor    
    correcction = 0.2
    measurment = float(line[3])
    # measurmaent for the center image
    sim_measurments.append(measurment)
    # measurmaent for the left image + correcction factor
    sim_measurments.append(measurment + correcction) 
    # measurmaent for the right image-  correcction factor
    sim_measurments.append(measurment - correcction)
# the image and measurments data set: 
# image = fliped_image +  simu_images
images =  []
measurments =  []
for image, measurment in zip(sim_images, sim_measurments):
    # flip the image about the verticle axes 
    fliped_image = cv2.flip(image,1)
    fliped_measurment =measurment*-1.0
    images.append(image)
    images.append(fliped_image)
    measurments.append(measurment)
    measurments.append(fliped_measurment)
    

# Keras accepts numpy file
X_train = np.array(images)
y_train = np.array(measurments)
print(len(X_train))
print(len(y_train))

# End to End Learning for Self-Driving Cars

model = Sequential()
# Normalization layer
model.add(Lambda(lambda x: x/ 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
# 5 x 5 Convolution filters + subsample which make the size of the images smaller
model.add(Convolution2D(24,5,5, subsample = (2,2), activation = 'relu', name='conv1'))
model.add(Convolution2D(36,5,5, subsample = (2,2), activation = 'relu', name='conv2'))
model.add(Convolution2D(48,5,5, subsample = (2,2), activation = 'relu', name='conv3'))
# 3 x 3 Convolution filters
model.add(Convolution2D(64,3,3,activation = 'relu', name='conv4'))
model.add(Convolution2D(64,3,3,activation = 'relu', name='conv5'))
model.add(Flatten())
model.add(Dropout(.8))
model.add(Dense(100))
model.add(Dropout(.8))
model.add(Dense(50)) 
model.add(Dense(10))
# one single output layer "Regression"
model.add(Dense(1))
# cost  function
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)
model.compile(optimizer=adam, loss = 'mse')
#model.fit(X_train,y_train, validation_split=0.2,shuffle=True, nb_epoch=3 )
#model.save('model.h5')