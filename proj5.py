# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.fftpack import fft, fftshift
# from suaBibSignal import *
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import soundfile   as sf
import peakutils

def calcFFT(signal, fs):
    # https://docs.scipy.org/doc/scipy/reference/tutorial/fftpack.html
    N  = len(signal)
    # W = window.hamming(N)
    T  = 1/fs
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    yf = fft(signal)
    return(xf, np.abs(yf[0:N//2]))

def generateSin(freq, amplitude, time, fs):
    n = time*fs
    x = np.linspace(0.0, time, n)
    s = amplitude*np.sin(freq*x*2*np.pi)
    return (x, s)

def plotFFT(signal, fs):
    x,y = calcFFT(signal, fs)
    plt.figure()
    plt.plot(x, np.abs(y))
    plt.title('Fourier')
    plt.show()
    return x,y

def rotinaUsuario():
    fs = 44100   # taxa de amostagem (sample rate)
    T  = 1
    t  = np.linspace(-T/2,T/2,T*fs)
    sd.default.samplerate = fs
    sd.default.channels = 1
    # true = True
    # while true:

    frequencias = []
    f_list = []
    y_list = []
    digitos = {
        697:  ["1","2","3","a"],
        770:  ["4","5","6","b"],
        852:  ["7","8","9","c"],
        941:  ["x","0","#","d"],
        1209: ["1","4","7","x"],
        1336: ["2","5","8","0"],
        1477: ["3","6","9","#"],
        1633: ["a","b","c","d"]
    }
    
    digito = input("Digite uma tecla:")
    if digito == "fim":
        true = False
    for i, e in digitos.items():
        if digito in e:
            frequencias.append(i)
            
    for i in frequencias:
        print(i)
        x, y = generateSin(i, 0.1, T, fs)
        f_list.append((x, y))
        y_list.append(y)
        
    for i, e in f_list:
        plt.plot(t, e, '.-')
        plt.xlim(0, 0.001)
        plt.show()
        sd.play(e)
        sd.wait()
    
    if len(y_list) != 0:
        f_sum =[sum(x) for x in zip(*y_list)]
        plt.plot(t, f_sum, '.-')
        plt.xlim(0, 0.001)
        plt.show()
        sd.play(f_sum)
        sd.wait()

        with open("freq.txt", "w") as file:
            file.write(str(f_sum))

def rotinaReceptor():
    fs = 44100   # taxa de amostagem (sample rate)
    T  = 1
    t  = np.linspace(0,T,T*fs)
    sd.default.samplerate = fs
    sd.default.channels = 1
    digitos = {
            697:  ["1","2","3","a"],
            770:  ["4","5","6","b"],
            852:  ["7","8","9","c"],
            941:  ["x","0","#","d"],
            1209: ["1","4","7","x"],
            1336: ["2","5","8","0"],
            1477: ["3","6","9","#"],
            1633: ["a","b","c","d"]
        }
    true = True

    with open("freq.txt", "r") as file:
        myrecording = file.read()
        myrecording = eval(myrecording)

    myrecording = sd.playrec(myrecording, fs, channels=1)
    sd.wait()
    plt.plot(t, myrecording, '.-')
    plt.title("Som do playrec")
    plt.show()
    
    sd.play(myrecording)
    sd.wait()
    
    X, Y = plotFFT(myrecording[:,0], fs)

    index = peakutils.indexes(np.abs(Y), thres = 0.2, min_dist=10)
    print("index de picos {}" .format(index))
    for freq in X[index]:
        print("freq de pico sao {}" .format(freq))

    freq1 = set(digitos[index[0]])
    freq2 = set(digitos[index[1]])
    caractere = freq1.intersection(freq2)

    print(f"O caractere digitado foi {list(caractere)[0]}")

rotinaUsuario()

rotinaReceptor()