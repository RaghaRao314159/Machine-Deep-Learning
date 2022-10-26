
from keras.models import Sequential 
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from keras.preprocessing.image import ImageDataGenerator


model = Sequential()
model.add(Conv2D(32,3,3,input_shape=(64,64,3),activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

training = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2, horizontal_flip = True).flow_from_directory("/Volumes/Ragha's Hard Drive/programming/training",target_size = (64, 64), batch_size = 32, class_mode = 'binary')
testing = ImageDataGenerator(rescale = 1./255).flow_from_directory("/Volumes/Ragha's Hard Drive/programming/testing",target_size = (64, 64),batch_size = 32,class_mode = 'binary')

model.fit_generator(training,steps_per_epoch = 1000,epochs = 2,validation_data = testing,validation_steps = 250)



#RESULTS------------------------
'''
Found 20000 images belonging to 2 classes.
Found 200 images belonging to 2 classes.
Epoch 1/2
1000/1000 [==============================] - 357s 357ms/step - loss: 0.5960 - accuracy: 0.6822 - val_loss: 0.6933 - val_accuracy: 0.6589
Epoch 2/2
1000/1000 [==============================] - 683s 683ms/step - loss: 0.5300 - accuracy: 0.7333 - val_loss: 0.4938 - val_accuracy: 0.7748'
'''
