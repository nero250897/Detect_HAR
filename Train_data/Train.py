import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot
import matplotlib.pyplot as plt

from keras.layers import LSTM, Dense, Dropout, Flatten
from keras.models import Sequential
from matplotlib import pyplot
from tensorflow import keras
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


# Doc du lieu dau vao
climb_df = pd.read_csv("CLIMB.txt")
utd_df = pd.read_csv("UNLOCK THE DOOR.txt")
walk_df = pd.read_csv("WALK.txt")

X = []
y = []

no_of_timesteps = 10

dataset = climb_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append([0,0,0])

dataset = utd_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append([1,0,0])

dataset = walk_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append([0,1,0])

X, y = np.array(X), np.array(y)
#print(X.shape, y.shape)

# Chia du lieu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model  = Sequential()
model.add(LSTM(units = 50, return_sequences = True, input_shape = (no_of_timesteps, X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
model.add(Dense(units = 3, activation="softmax"))
model.compile(optimizer=Adam(learning_rate=0.0001), metrics = ['accuracy'], loss = "binary_crossentropy")
model.summary() # in ra kien truc mang
history = model.fit(X_train, y_train, epochs=16, batch_size=32, validation_data=(X_test, y_test))

y_hat = model.predict(X_test)
y_pred = np.argmax(y_hat, axis=1)
y_test_label =  np.argmax(y_test, axis=1)

model.save("model_101.h5")

# Tính accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test_label, y_pred)
print('Accuracy: %f' % accuracy)
# Tính precision tp / (tp + fp)
precision = precision_score(y_test_label, y_pred, average='macro')
print('Precision: %f' % precision)
# Tính recall: tp / (tp + fn)
recall = recall_score(y_test_label, y_pred, average='macro')
print('Recall: %f' % recall)
# Tính f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test_label, y_pred, average='macro')
print('F1 score: %f' % f1)
print("=============================")
print("=============================")


fig, axs = plt.subplots(1, 2, figsize=(15, 5))
axs[0].plot(history.history['accuracy'])
axs[0].plot(history.history['val_accuracy'])
axs[0].set_title('model accuracy')
axs[0].set_ylabel('accuracy')
axs[0].set_xlabel('epoch')
axs[0].legend(['train', 'Vadidation'])
axs[1].plot(history.history['loss'])
axs[1].plot(history.history['val_loss'])
axs[1].set_title('model loss')
axs[1].set_ylabel('loss')
axs[1].set_xlabel('epoch')
axs[1].legend(['train', 'Vadidation'])
matplotlib.pyplot.show()

#print(matrix)
#matrix = confusion_matrix(y_test_label, y_pred)