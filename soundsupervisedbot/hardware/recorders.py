import os
import pyaudio
import wave
import threading
import torch, torchvision
import numpy as np

class MicRecorder():
    '''
    Modified from : https://gist.github.com/kepler62f/9d5836a1eff8b372ddf6de43b5b74d95
    A recorder class for recording audio stream from a microphone to WAV files.
    Uses non-blocking callback threads to get audio stream but uses a list
    to save chunks of stream to file
    output_path: string, folder to output wave files
    channels: integer, 1 mono, 2 stereo
    rate: integer, microphone sampling rate (hertz)
    frames_per_buffer: integer
    Example:
        from micrecorder import MicRecorder
        
        rec = MicRecorder('./audio-clips', overlap=2)
        rec.start_recording()
    '''

    def __init__(self, output_path, channels=1, rate=16000, frames_per_buffer=1024, input_device_index=None):
        self.output_path = output_path
        self.channels = channels
        self.rate = rate
        self.frames_per_buffer = frames_per_buffer

        self._pa = pyaudio.PyAudio()
        self._stream = None
        self.frames = []
        self.input_device_index = input_device_index
        
    def stop_recording(self):
        self._stream.stop_stream()
        self._stream.close()
        self._pa.terminate()
        
    def get_callback(self):
        def callback(in_data, frame_count, time_info, status):
            self.frames.append(in_data)
            return in_data, pyaudio.paContinue
        return callback
    
    def write_audio_file(self, mode='wb', clear_frames=True):
        
        filename = os.path.join(self.output_path, 'audio-clip.wav')
        
        wavefile = wave.open(filename, mode)
        wavefile.setnchannels(self.channels)
        wavefile.setsampwidth(self._pa.get_sample_size(pyaudio.paInt16))
        wavefile.setframerate(self.rate)
        wavefile.writeframes(b''.join(self.frames))
        wavefile.close()

        if clear_frames:
            del self.frames
            self.frames = []

        print("Saved {}".format(filename))
           
    def _start_stream(self):
        self._stream = self._pa.open(format=pyaudio.paInt16,
                                     channels=self.channels,
                                     rate=self.rate,
                                     input=True,
                                     input_device_index = self.input_device_index,
                                     frames_per_buffer=self.frames_per_buffer,
                                     stream_callback=self.get_callback())
        print('Begin recording...')
        self._stream.start_stream()


class ImageRecorder(threading.Thread):

    def __init__(self, cam= None,save_dir = None):
        super().__init__()
        self.cam = cam
        self.save_dir = save_dir
        self.exit = False
        self.imgs = []

    def run(self):
        while not self.exit:
            bgr, dep = self.cam.get_frame()
            self.imgs.append(torch.Tensor(np.array(bgr)))

    def stop(self):
        self.exit = True
        self.join()
    
    def save_video(self):
        torchvision.io.write_video(str(os.path.join(self.save_dir, 'video.mp4')), torch.stack(self.imgs), 30)

    def clear(self):
        del self.imgs
        self.imgs = []
