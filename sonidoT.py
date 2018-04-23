import math
import numpy as np
import pyaudio
import wave

def sine(frequency, length, rate):
    length = int(length * rate)
    factor = float(frequency) * (math.pi * 2) / rate
    return np.sin(np.arange(length) * factor)


def play_tone(stream, frequency, length, rate=44100):
    chunks = []
    chunks.append(sine(frequency, length, rate))

    chunk = np.concatenate(chunks) * 0.25

    stream.write(chunk.astype(np.float32).tostring())


#if __name__ == '__main__':
def sonidoUnico():
    
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,
                        channels=1, rate=44100, output=1)
    for m in list(range(10)):
        play_tone(stream, 1000,0.1)
        play_tone(stream, 7000,0.1)

    stream.close()
    p.terminate()


def play_wave(stream, wave):
    chunks = []
    chunks.append(wave)
    chunk = np.concatenate(chunks)*0.25
    stream.write(chunk.astype(np.float32).tostring())


def play_soundS():
    #wave, A = wavefunc(t, freq, A=A)
    #S = sigmoid(t)
    #signal = wave*S

    signal = sine(1040, 5,44100)
    signal2 = sine(40, 5,44100)
    #stereo_signal = np.zeros([len(signal), 2])   #these two lines are new
    #stereo_signal = np.zeros((len(signal))) 
    #stereo_signal[:, 1] = signal[:]     #1 for right speaker, 0 for  left
    #stereo_signal = [signal, stereo_signal]

    stereo_signal = np.ravel(np.column_stack((signal,signal2)))

    p = pyaudio.PyAudio()
    stream = p.open(channels=2, 
                rate=44100, 
                format=pyaudio.paFloat32, 
                output=True)
    play_wave(stream,stereo_signal)
    #play_wave(stream,signal2)
    #stream.write(stereo_signal.tostring())
    stream.close()
    p.terminate()


#sonidoUnico()
play_soundS()