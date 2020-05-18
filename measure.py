import pyaudio
import numpy as np
import signal
import sys
import time
import threading

CHUNK = 4096
RATE = 119200
FREQUENCY = 14000
FREQUENCY_A = 16000
FREQUENCY_B = 12000
THRESHOLD = 100000
VOLUME = 1
DURATION = 0.1

p = pyaudio.PyAudio()

# both await A0
AWAIT_A0 = 0
# only A await B0
AWAIT_B0 = 1
# only B awaits A1
AWAIT_A1 = 2

MODE_A = 0
MODE_B = 1

t0 = None
t1 = None

def play_tone():
    out_stream.write(1*samples)

def monitor(mode):
    global t0
    global t1
    status = AWAIT_A0 if mode == MODE_B else AWAIT_B0
    mic_stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True,
      frames_per_buffer=CHUNK)
    while True:
        data = np.frombuffer(mic_stream.read(CHUNK),dtype=np.int16)
        fft = abs(np.fft.fft(data).real)
        fft = fft[:int(len(fft)/2)]
        freq = np.fft.fftfreq(CHUNK,1.0/RATE)
        freq = freq[:int(len(freq)/2)]
        assert freq[-1]>FREQUENCY, "ERROR: increase chunk size"
        if mode == MODE_A:
            val = fft[np.where((freq>FREQUENCY_B) & (freq<FREQUENCY_A))[0][0]]
        else:
            val = fft[np.where(freq>FREQUENCY_A)[0][0]]
        if val > THRESHOLD:
            if status == AWAIT_A0:
                # we are B: A has messaged us first
                play_tone()
                t0 = time.time()
                status = AWAIT_A1
            elif status == AWAIT_B0:
                # we are A: B has responded to us
                t1 = time.time()
                play_tone()
                break
            elif status == AWAIT_A1:
                # we are B: A has messaged us again
                t1 = time.time()
                break
            else:
                break

    mic_stream.stop_stream()
    mic_stream.close()


if len(sys.argv) < 2:
    print('please provide \'a\' or \'b\'')
    exit(-1)

if sys.argv[1] == 'a':
    out_freq = FREQUENCY_A
    mode = MODE_A
elif sys.argv[1] == 'b':
    out_freq = FREQUENCY_B
    mode = MODE_B
else:
    print('only provide \'a\' or \'b\'')
    exit(-1)

listen_thread = threading.Thread(target=monitor, args=[mode])
listen_thread.start()

samples = (np.sin(2*np.pi*np.arange(RATE*DURATION)*out_freq/RATE)).astype(np.float32)
out_stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=RATE,
                output=True)


if mode == MODE_A:
    input("Enter to begin")
    play_tone()
    t0 = time.time()
else:
    print('I am device B')

listen_thread.join()

out_stream.stop_stream()
out_stream.close()

p.terminate()

sos = 343.0
rtt = t1 - t0
print(rtt)
print((rtt) * (sos/2))