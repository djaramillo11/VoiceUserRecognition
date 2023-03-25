import numpy
import numpy as np
import pyaudio
import wave
from sys import byteorder
from array import array
from struct import pack
from utils import extract_feature

"Settings to record audio"
THRESHOLD = 500
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 16000
SILENCE = 30

"Flag to indicate if user is new (False) or already registered (true)"
isRegistered=True

"data [user][feature n]"
data =[]
"Value from feature n1, n2, n3..."
user_feature={0.5,0.2}

"If it is a new user, take his\her name"
userName="David"

def is_silent(snd_data):
    "Returns 'True' if below the 'silent' threshold"
    return max(snd_data) < THRESHOLD

def normalize(snd_data):
    "Average the volume out"
    MAXIMUM = 16384
    times = float(MAXIMUM)/max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i*times))
    return r

def trim(snd_data):
    "Trim the blank spots at the start and end"
    def _trim(snd_data):
        snd_started = False
        r = array('h')

        for i in snd_data:
            if not snd_started and abs(i)>THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r

    # Trim to the left
    snd_data = _trim(snd_data)

    # Trim to the right
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data

def add_silence(snd_data, seconds):
    "Add silence to the start and end of 'snd_data' of length 'seconds' (float)"
    r = array('h', [0 for i in range(int(seconds*RATE))])
    r.extend(snd_data)
    r.extend([0 for i in range(int(seconds*RATE))])
    return r


def record():
    """
    Record a word or words from the microphone and
    return the data as an array of signed shorts.

    Normalizes the audio, trims silence from the
    start and end, and pads with 0.5 seconds of
    blank sound to make sure VLC et al can play
    it without getting chopped off.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
        input=True, output=True,
        frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    snd_started = False

    r = array('h')

    while 1:
        # little endian, signed short
        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

        silent = is_silent(snd_data)

        if silent and snd_started:
            num_silent += 1
        elif not silent and not snd_started:
            snd_started = True

        if snd_started and num_silent > SILENCE:
            break

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    r = normalize(r)
    r = trim(r)
    r = add_silence(r, 0.5)
    return sample_width, r

def record_to_file(path):
    "Records from the microphone and outputs the resulting data to 'path'"
    sample_width, data = record()
    data = pack('<' + ('h'*len(data)), *data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()

def rms_values(filename):
    "Following features can be included in the below function: mfcc,chroma,mel,contrast,tonnetz,rms,zcr"
    mfcc = extract_feature(filename, mfcc=True).reshape(1, -1)
    chroma = extract_feature(filename, chroma=True).reshape(1, -1)

    mfcc_rms = np.sqrt(np.mean(mfcc ** 2))
    chroma_rms = np.sqrt(np.mean(chroma ** 2))

    return mfcc_rms,chroma_rms

def identify_user(mfcc_rms,chroma_rms):
    file = open('database.txt','r')
    readFile = file.readlines()
    for line in readFile:
        user = [line.strip() for line in line.split(' ')]
        data.append(user)
    #print("data", data[user-row][feature-columns])
    userFound = False
    length = len(data)
    for i in range(length):
        user_search = np.isclose(mfcc_rms,float(data[i][1]), rtol=300e-3, atol=300e-03)
        if user_search:
            user=data[i][0]
            userFound=True
            break;
    if not userFound:
        user = "User not Found"
    return user

def register_user(userName,mfcc_rms,chroma_rms):
    with open('database.txt', 'a') as f:
        f.write(userName+" "+ str(mfcc_rms)+" "+str(chroma_rms)+'\n')
        f.close()
    print("User saved as:",userName)

if __name__ == "__main__":

    print("Please talk")
    filename = "test.wav"
    record_to_file(filename)
    mfcc_rms,chroma_rms=rms_values(filename)
    if isRegistered:
        user=identify_user(mfcc_rms,chroma_rms)
        print("User found as",user)
    else:
        register_user(userName,mfcc_rms,chroma_rms)
