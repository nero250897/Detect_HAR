import numpy as np
import pandas as pd
import matplotlib.pyplot
import time
import tensorboard
import tensorflow as tf

from keras.layers import LSTM, Dense,Dropout, BatchNormalization
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score



# Đọc dữ liệu
#walk_df = pd.read_csv("WALK.txt")
climb_df = pd.read_csv("CLIMB.txt")
utd_df = pd.read_csv("UNLOCK THE DOOR.txt")

X = []
y = []
no_of_timesteps = 10

dataset = climb_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append(0)

dataset = utd_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append(1)

X, y = np.array(X), np.array(y)
print(X.shape, y.shape)

#Chia du lieu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

model  = Sequential()
model.add(LSTM(128, return_sequences = True, activation='relu', input_shape = (no_of_timesteps, X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(256, return_sequences= True, activation='relu'))
model.add(Dropout(0.2))
model.add(LSTM(256, return_sequences= True, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(units=3, activation="softmax"))
model.summary()
model.compile(optimizer=Adam('learning_rate =0.001'), metrics = ['categorical_accuracy'], loss = "categorical_crossentropy")
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
model.save("model.h5")

matplotlib.pyplot.plot(history.history['categorical_accuracy'])
matplotlib.pyplot.plot(history.history['val_categorical_accuracy'])
matplotlib.pyplot.title('model accuracy')
matplotlib.pyplot.ylabel('accuracy')
matplotlib.pyplot.xlabel('epoch')
matplotlib.pyplot.legend(['train', 'Vadidation'])
matplotlib.pyplot.show()




