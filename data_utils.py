import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from scipy import signal
import scipy.io.wavfile as wav
import librosa
import random
from sklearn import preprocessing
import tensorflow as tf

_n_fft = 512
_hop_length = 256

def cal_minmax(file_list):
    data = np.zeros(((_n_fft/2)+1, 3000000), dtype=np.float32)
    cnt_idx = 0
    for filename in file_list:
        sr, y = wav.read(filename)
        if sr != 16000:
            raise ValueError('Sampling rate is expected to be 16kHz!')
        if y.dtype!='float32':
            y = np.float32(y/32767.)

        D=librosa.stft(y,n_fft=_n_fft,hop_length=_hop_length,win_length=_n_fft,window=scipy.signal.hamming)
        Sxx=np.log10(abs(D)**2) 
        data[:, cnt_idx:cnt_idx+Sxx.shape[1]] = Sxx
        cnt_idx += Sxx.shape[1]

    _max = np.max(data, axis=1).reshape(((_n_fft/2)+1, 1))
    _min = np.min(data, axis=1).reshape(((_n_fft/2)+1, 1))      
    return _max, _min

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

class dataPreprocessor(object):
    def __init__(self, record_path, record_name, noisy_filelist=None, clean_filelist=None, use_waveform=True):
        self.noisy_filelist = noisy_filelist
        self.clean_filelist = clean_filelist
        self.use_waveform = use_waveform
        self.record_path = record_path
        self.record_name = record_name

        if use_waveform:
            self.FRAMELENGTH = 16384
            self.OVERLAP = self.FRAMELENGTH//2
        else:
            self.FRAMELENGTH = 8*8
            self.OVERLAP = self.FRAMELENGTH//16

        if noisy_filelist and clean_filelist:
            print('Write records')
        else:
            print('Read-only mode')

    def slice_signal(self, signal, window_size, overlap):
        """ Return windows of the given signal by sweeping in stride fractions
            of window
        """
        n_samples = signal.shape[0]
        offset = overlap
        slices = []
        for beg_i, end_i in zip(range(0, n_samples, offset),
                                range(window_size, n_samples + offset,
                                      offset)):
            if end_i - beg_i < window_size:
                break
            slice_ = signal[beg_i:end_i]
            if slice_.shape[0] == window_size:
                slices.append(slice_)
        return np.array(slices, dtype=np.int32)

    def read_and_slice(self, filename):
        sr, wav_data = wav.read(filename)
        print(wav_data)
        if sr != 16000:
            raise ValueError('Sampling rate is expected to be 16kHz!')
        signals = self.slice_signal(wav_data, self.FRAMELENGTH, self.OVERLAP)
        return signals  

    def make_spectrum(self, filename, normalize_mode=None, _max=None, _min=None):
        sr, y = wav.read(filename)
        if sr != 16000:
            raise ValueError('Sampling rate is expected to be 16kHz!')
        if y.dtype!='float32':
            y = np.float32(y/32767.)

        D=librosa.stft(y,n_fft=_n_fft,hop_length=_hop_length,win_length=_n_fft,window=scipy.signal.hamming)
        Sxx=np.log10(abs(D)**2) 
        if normalize_mode=='mean_std':
            mean = np.mean(Sxx, axis=1).reshape(((_n_fft/2)+1, 1))
            std = np.std(Sxx, axis=1).reshape(((_n_fft/2)+1, 1))+1e-12
            Sxx = (Sxx-mean)/std  
        elif normalize_mode=='minmax':
            Sxx = 2 * (Sxx - _min)/(_max - _min) - 1
        print(np.max(Sxx), np.min(Sxx))
        slices = []
        for i in range(0, Sxx.shape[1]-self.FRAMELENGTH, self.OVERLAP):
            # frequency bin = 256d
            slices.append(Sxx[1:,i:i+self.FRAMELENGTH]) 
        return np.array(slices)   

    def write_tfrecord(self):
        if self.noisy_filelist is None or self.clean_filelist is None:
            raise ValueError('Read-only mode\nCreate an instance by specifying a filename.')

        if tf.gfile.Exists(self.record_path):
            print('Folder already exists: {}\n'.format(self.record_path))
        else:
            tf.gfile.MkDir(self.record_path)

        nlist = [x[:-1] for x in open(self.noisy_filelist).readlines()]
        clist = [x[:-1] for x in open(self.clean_filelist).readlines()]

        assert len(nlist) == len(clist)

        out_file = tf.python_io.TFRecordWriter(self.record_path+self.record_name)

        if self.use_waveform:
            for n,c in zip(nlist, clist):
                print(n,c)
                wav_signals = self.read_and_slice(c)
                noisy_signals = self.read_and_slice(n)
                print(wav_signals)            
                for (wav, noisy) in zip(wav_signals, noisy_signals):
                    wav_raw = wav.tostring()
                    noisy_raw = noisy.tostring()
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'wav_raw': _bytes_feature(wav_raw),
                        'noisy_raw': _bytes_feature(noisy_raw)}))
                    out_file.write(example.SerializeToString())
            out_file.close()

        else:
            # Calculate min and max of entire training data
            _max, _min = cal_minmax(clist)
            for n,c in zip(nlist, clist):
                print(n,c)
                wav_signals = self.make_spectrum(c, 'minmax', _max, _min)
                noisy_signals = self.make_spectrum(n, 'mean_std')
                for (wav, noisy) in zip(wav_signals, noisy_signals):
                    print(wav.shape, noisy.shape)
                    wav_raw = wav.tostring()
                    noisy_raw = noisy.tostring()
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'wav_raw': _bytes_feature(wav_raw),
                        'noisy_raw': _bytes_feature(noisy_raw)}))
                    out_file.write(example.SerializeToString())
            out_file.close()             

    def read_and_decode(self,batch_size=16, num_threads=8):
        filename_queue = tf.train.string_input_producer([self.record_path+self.record_name])        
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
                serialized_example,
                features={
                    'wav_raw': tf.FixedLenFeature([], tf.string),
                    'noisy_raw': tf.FixedLenFeature([], tf.string),
                })
        if self.use_waveform:
            wave = tf.decode_raw(features['wav_raw'], tf.int32)
            wave.set_shape(self.FRAMELENGTH)
            wave = (2./65535.) * tf.cast((wave - 32767), tf.float32) + 1.
            wave = tf.reshape(wave, [1,-1,1])
            noisy = tf.decode_raw(features['noisy_raw'], tf.int32)
            noisy.set_shape(self.FRAMELENGTH)
            noisy = (2./65535.) * tf.cast((noisy - 32767), tf.float32) + 1.
            noisy = tf.reshape(noisy, [1,-1,1])
        else:
            wave = tf.decode_raw(features['wav_raw'], tf.float32)
            wave = tf.reshape(wave,[1,256,self.FRAMELENGTH])
            noisy = tf.decode_raw(features['noisy_raw'], tf.float32)
            noisy = tf.reshape(noisy,[1,256,self.FRAMELENGTH])

        wavebatch, noisybatch = tf.train.shuffle_batch([wave,
                                             noisy],
                                             batch_size=batch_size,
                                             num_threads=num_threads,
                                             capacity=1000 + 3 * batch_size,
                                             min_after_dequeue=1000,
                                             name='wav_and_noisy')
        return wavebatch, noisybatch
 
