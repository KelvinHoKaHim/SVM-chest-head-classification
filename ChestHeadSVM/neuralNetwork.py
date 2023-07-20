from scipy.io import wavfile
from scipy.fft import fft
import numpy as np
import pandas as pd
from librosa.effects import preemphasis
from matplotlib import pyplot as plt
import tensorflow as tf
import sys

def plot(dataSet):
    colour = ["red","blue",]
    for item in ["H","r1","r2","K","SC",'SS',"E"]:
        graph = [[],[],[]]
        for i in range(len(dataSet[item])):
            graph[dataSet["CH"][i]].append(dataSet[item][i])
        for i in range(len(colour)):
            plt.hist(graph[i], color = colour[i], alpha = 0.5, bins=100)
        plt.title(item)
        plt.savefig("ChestHeadSVM/" + item)
        plt.clf()
        for i in range(len(colour)):
            plt.hist(graph[i], color = colour[i], alpha = 0.5, bins=100)
            plt.title(item)
            plt.savefig("ChestHeadSVM/" + item + colour[i])
            plt.clf()


def preprocessing(soundTrack):
    soundTrack = soundTrack - np.average(soundTrack)
    soundTrack = preemphasis(soundTrack)
    soundTrack = soundTrack*np.hamming(len(soundTrack))
    return soundTrack

def cut(soundTrack):
    length = 18.860408
    cuts = list(map(float, "1.594575 3.548696 4.315016 6.537345 7.418613 10.522211 11.595059 14.123916".split()))
    print("There are" , len(cuts) + 1 , " fragments. What are their respective sound register?")
    chestHeadPos = list(map(int, "2 0 2 1 2 0 2 1 2".split()))
    if len(chestHeadPos) == len(cuts) + 1:
        cuts.append(length)
        tracks = []
        last = 0
        for i in range(len(chestHeadPos)):
            if chestHeadPos[i] != 2:
                tracks.append((soundTrack[int(last*len(soundTrack)/length):int(cuts[i]*len(soundTrack)/length)], chestHeadPos[i]))
            last = cuts[i]
        return tracks
    else:
        print("error")
        return None
    
def makeTensor(cuts):
    X = np.empty((0,2048))
    y = np.empty((0))
    for cut in cuts:
        track = cut[0]
        chestHead = cut[1]
        for i in range(0,(len(track)//2048)*2048,2048):
            X = np.vstack((X, (lambda arr : arr / sum(arr))(list(map(lambda a : np.abs(a)**2, fft(track[i : i+2048]))))))
            y = np.append(y, chestHead)
    print(X.shape, y.shape)
    print(y)
    return (X,y)


rate, soundTrack= wavfile.read("ChestHeadSVM/ChestHead_Sample_01_John.wav")
track = preprocessing(soundTrack.T[0])
track = cut(track)
X , y = makeTensor(track)

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(2048,)),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])
model.compile(optimizer='adam', loss="binary_crossentropy", metrics=['accuracy'])
print("Compiling completed")
model.fit(X,y, epochs=10)
print("Training completed")
_, accuracy = model.evaluate(X, y)
print("========================================")
print('Accuracy:', accuracy)

result = model.predict(X)
for i in range(len(y)):
    print(result[i], y[i])

rate, soundTrack= wavfile.read("ChestHeadSVM/ChestHead_Sample_03_Clipped_spencerwelIG.wav")
soundTrack = preprocessing(soundTrack.T[0])
X_new = np.empty((0,2048))
for i in range(0,(len(soundTrack)//2048)*2048, 2048):
    X_new = np.vstack((X_new, (lambda arr : arr / sum(arr))(list(map(lambda a : np.abs(a)**2, fft(soundTrack[i:i+2048]))))))
result = model.predict(X_new)
for elem in result:
    print(elem, np.argmax(elem))