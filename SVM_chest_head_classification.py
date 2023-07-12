from scipy.io import wavfile
from scipy.fft import fft
import numpy as np
import pandas as pd
from librosa.effects import preemphasis
from matplotlib import pyplot as plt


def autoCorrelation(signal):
    ac = []
    avg = np.average(signal)
    for k in range(len(signal)):
        val = 0
        for i in range(len(signal)-k):
            val += (signal[i]-avg)*(signal[i+k]-avg)
        ac.append(val)
    return ac

def toSpectorgramAndPSD(signal, freq, low, high): #generate spectrogram and PSD
    cut = lambda arr : arr[int(low*len(arr)/freq): int(high*len(arr)/freq)]
    signal = preemphasis(signal - np.average(signal)) * np.hamming(len(signal))
    spec = (list(map(lambda a : np.abs(a)**2,fft(signal)[int(low*len(signal)/freq):int(high*len(signal)/freq)])))
    #print(len(spec), len(autoCorrelation(spec)))
    return ((lambda arr : arr / sum(arr))(cut(np.log10(spec))), list(map(lambda a : np.abs(a)**2, fft(autoCorrelation(spec)))))

def features(signal, freq, high, low): #feature Extractions
    spectrogram, PSD = toSpectorgramAndPSD(signal, freq, low, high)
    def harmonicity():
        return max(autoCorrelation(spectrogram)[1:])
    def SPD():
        cut100 = int(len(PSD)*100/freq)
        cut1000 = int(len(PSD)*1000/freq)
        cut2000 = int(len(PSD)*2000/freq)
        cut6000 = int(len(PSD)*6000/freq)
        #print("SPC", sum(SPDCell[cut2000:cut6000 + 1])/sum(SPDCell[cut100:cut2000 + 1]), sum(SPDCell[cut1000:cut6000 + 1])/sum(SPDCell[cut100:cut1000 + 1]))
        return (sum(PSD[cut2000:cut6000 + 1])/sum(PSD[cut100:cut2000 + 1]), sum(PSD[cut1000:cut6000 + 1])/sum(PSD[cut100:cut1000 + 1]))
    def kurtosis():
        avg = np.average(spectrogram)
        numerator = sum(map(lambda a : (a-avg)**4 , spectrogram))
        denominator = sum(map(lambda a : (a-avg)**2 , spectrogram))**2
        return len(spectrogram)*numerator/denominator
    def centroidAndSpread(low, high):
        stepSize = freq / len(PSD)
        low = int(len(PSD)*low/freq)
        high = int(len(PSD)*high/freq)
        nominator = 0
        denominator = sum(PSD[low:high+1])
        for i in range(low,high+1):
            nominator += np.log2(stepSize*(i+0.5)/1000)*PSD[i]
        SC = nominator/denominator
        nominator = 0
        for i in range(low,high+1):
            nominator += ((np.log2(stepSize*(i+0.5)/1000)-SC)**2)*PSD[i]
        SS = np.sqrt(nominator/denominator)
        #print("SC SS", SC, SS)
        return (SC, SS)
    r1 , r2 = SPD()
    SC , SS = centroidAndSpread(low,high)
    return (harmonicity(), r1, r2, kurtosis(), SC, SS, sum(PSD))

def toCSV(soundTrack, freqMax, length):
    data = {
        "H" : [],
        "r1" : [],
        "r2" : [],
        "K" : [],
        "SC" : [],
        "SS" : [],
        "E" : [],
        "CH" : []
    }
    for sound in soundTrack:
        track = sound[0]
        chestHead = sound[1]
        for i in range(0,len(track), 2048):
            curr = track[i:i+2048]
            H, r1, r2, K, SC, SS, E = features(curr, 44100, 10000, 60)
            data["H"].append(H)
            data["r1"].append(r1)
            data["r2"].append(r2)
            data["K"].append(K)
            data["SC"].append(SC)
            data["SS"].append(SS)
            data["E"].append(E)
            data["CH"].append(chestHead)
            #print(data["H"][-1], data["r1"][-1], data["r2"][-1], data["K"][-1], data["SC"][-1], data["SS"][-1], data["E"][-1], data["CH"][-1])
    df = pd.DataFrame(data)
    df.to_csv('ChestHeadSVM/ChestHeadData.csv', mode='a', index=False, header=True)
    return data


def plot(dataSet): #plot graphs
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

def cut(soundTrack): #cut sound track into several segment, each with their coreesponding chestHead value (0 = chest, 1 = head, 2 = noise)
    length = 18.860408
    cuts = list(map(float, "1.594575 3.548696 4.315016 6.537345 7.418613 10.522211 11.595059 14.123916".split())) #the exact time when a transition occur
    print("There are" , len(cuts) + 1 , " segment. What are their respective sound register?")
    chestHeadPos = list(map(int, "2 0 2 1 2 0 2 1 2".split()))#if there are n transition, then the sound track can be cutted into n+1 segments. Hence there are n+1 labels
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
    


rate, soundTrack= wavfile.read("ChestHeadSVM/ChestHead_Sample_01_John.wav")
track = cut(soundTrack.T[0])
data = toCSV(track, rate, 18.860408)# the full length of the sound track
plot(data)
    

