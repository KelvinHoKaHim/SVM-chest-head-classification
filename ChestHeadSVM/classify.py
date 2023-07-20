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

def parameters(signal, maxFreq):
    spectrogram = list(map(lambda a : np.abs(a)**2, fft(signal)))
    spectrogram /= sum(spectrogram)
    logSpec = np.log10(spectrogram)
    logSpec /= sum(logSpec)
    #auto = autoCorrelation(spectrogram)
    H = max(autoCorrelation(logSpec[1:]))*10000
    PSD = list(map(lambda a : np.abs(a)**2, fft(autoCorrelation(spectrogram))))
    E = sum(PSD)
    def SPDRatio(SPDCell, maxFreq):
        cut100 = int(len(SPDCell)*100/maxFreq)
        cut1000 = int(len(SPDCell)*1000/maxFreq)
        cut2000 = int(len(SPDCell)*2000/maxFreq)
        cut6000 = int(len(SPDCell)*6000/maxFreq)
        #print("SPC", sum(SPDCell[cut2000:cut6000 + 1])/sum(SPDCell[cut100:cut2000 + 1]), sum(SPDCell[cut1000:cut6000 + 1])/sum(SPDCell[cut100:cut1000 + 1]))
        return (sum(SPDCell[cut2000:cut6000 + 1])/sum(SPDCell[cut100:cut2000 + 1]), sum(SPDCell[cut1000:cut6000 + 1])/sum(SPDCell[cut100:cut1000 + 1]))
    def kurtosis(cell ,low, high, freq):
        lowPos = int(len(cell)*low/freq)
        highPos = int(len(cell)*high/freq)
        avg = np.average(cell)
        denominator = 0
        nominator = 0
        for sample in cell[lowPos:highPos+1]:
            nominator += (sample - avg)**4
            denominator += (sample - avg)**2
        #print("kurt", len(ceil)*nominator/denominator)
        return (highPos-lowPos+1)*nominator/denominator**2
    def spectrumCentroidAndSpead(PSDCEil, maxFreq, lowFreq, hiFreq):
        stepSize = maxFreq / len(PSDCEil)
        low = int(len(PSD)*lowFreq/maxFreq)
        high = int(len(PSD)*hiFreq/maxFreq)
        nominator = 0
        denominator = sum(PSDCEil[low:high+1])
        for i in range(low,high+1):
            nominator += np.log2(stepSize*(i+0.5)/1000)*PSDCEil[i]
        SC = nominator/denominator
        nominator = 0
        for i in range(low,high+1):
            nominator += ((np.log2(stepSize*(i+0.5)/1000)-SC)**2)*PSDCEil[i]
        SS = np.sqrt(nominator/denominator)
        #print("SC SS", SC, SS)
        return (SC, SS)
    r1, r2 = SPDRatio(PSD, maxFreq)
    K = kurtosis(logSpec, 100, 10000, 44100)
    SC , SS = spectrumCentroidAndSpead(PSD, maxFreq, 100, 10000)
    return (H, r1, r2, K, SC, SS, E)

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
            H, r1, r2, K, SC, SS, E = parameters(curr, 44100)
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
    soundTrack -= np.average(soundTrack)
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
    


rate, soundTrack= wavfile.read("ChestHeadSVM/ChestHead_Sample_01_John.wav")
track = preprocessing(soundTrack.T[0])
track = cut(track)
data = toCSV(track, rate, 18.860408)
plot(data)
