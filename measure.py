import pyaudio
import numpy as np
import threading
from scipy.signal import correlate, find_peaks
import time
from itertools import takewhile

RATE = 44100

COS_FREQUENCY = 2000
COS_DURATION = 0.05

FREQUENCY = 6000
DURATION = 0.05
NUM_SAMPLES = RATE * DURATION
CHUNK = 4096

p = pyaudio.PyAudio()

sin_samples = (np.sin(2*np.pi*np.arange(NUM_SAMPLES)*FREQUENCY/RATE)).astype(np.float32)
cos_samples = (np.cos(2*np.pi*np.arange(RATE*COS_DURATION)*COS_FREQUENCY/RATE)).astype(np.float32)
out_stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=RATE,
                output=True)

def chirp():
    print('CHIRP!')
    out_stream.write(cos_samples)
    out_stream.write(sin_samples)
    last_chirp = time.time()

PEAK_WIDTH = 100
SD_THRESHOLD = 500
MP_THRESHOLD = 0.85

def monitor():
    mic_stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True,
      frames_per_buffer=CHUNK)
    prev = np.frombuffer(mic_stream.read(CHUNK),dtype=np.int16)
    peaks = []
    idx = 0
    while True:
        data = np.frombuffer(mic_stream.read(CHUNK),dtype=np.int16)

        crossed = list(correlate(data, sin_samples))
        peak_index = crossed.index(max(crossed))

        shadow_peaks = list(takewhile(lambda i: i != peak_index, list(find_peaks(crossed)[0])))


        if len(peaks) > 0 and peak_index + idx - peaks[-1] < RATE:
            # TODO: replace? indoor shadowing effect
            idx += CHUNK
            continue

        white_window = prev[:PEAK_WIDTH]

        if peak_index + (PEAK_WIDTH/2) >= len(crossed):
            peak_window = crossed[-PEAK_WIDTH:]
        elif peak_index - (PEAK_WIDTH/2) < 0:
            peak_window = crossed[:PEAK_WIDTH]
        else:
            peak_window = crossed[int(peak_index-(PEAK_WIDTH/2)):int(peak_index+(PEAK_WIDTH/2))]

        l2_peak = np.linalg.norm(peak_window)
        l2_white = np.linalg.norm(white_window)

        if l2_white != 0 and l2_peak / l2_white > SD_THRESHOLD:
            if len(peaks) < 2:
                if shadow_peaks != []:
                    sharp_record = 0
                    sharpness = {}
                    for shadow_peak in shadow_peaks:
                        sp_window = crossed[max(0,int(shadow_peak - PEAK_WIDTH/2)):min(len(crossed)-1,int(shadow_peak + PEAK_WIDTH/2))]
                        sharpness[shadow_peak] = crossed[shadow_peak] / (sum(sp_window) / len(sp_window))
                    max_sharpness = max(sharpness.values())
                    for shadow_peak in shadow_peaks:
                        if sharpness[shadow_peak] >= max_sharpness * MP_THRESHOLD:
                            peak_index = shadow_peak
                            break
                peaks.append(idx+peak_index)
                if len(peaks) == 2:
                    print(peaks, peaks[1] - peaks[0])


        idx += CHUNK
        prev = data


    mic_stream.stop_stream()
    mic_stream.close()

listen_thread = threading.Thread(target=monitor, args=[])
listen_thread.start()

input("Enter to chirp")
chirp()

listen_thread.join()

out_stream.stop_stream()
out_stream.close()

p.terminate()

sos = 343.0
print('a')