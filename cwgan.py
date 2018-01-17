from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import numpy as np
from ops import *
from data_utils import *
import re

class GradientPenaltyWGAN(object):
    '''
    Wasserstein GAN with Gradient Penalty (conditional version)
    '''
    def __init__(self, g_net, d_net, 
                data_noise, data_clean, 
                log_path, model_path, 
                use_waveform, 
                lr=1e-4, gan_lamb=1.):

        self.model_path = model_path
        self.log_path = log_path

        self.lamb_gp = 10.0
        self.lamb_recon = 1.0
        self.gan_lamb = gan_lamb
        self.z_dim = 128
        self.lr = lr
        self.g_net = g_net
        self.d_net = d_net
        self.real_noise = data_noise
        self.real_clean = data_clean

    def w_loss(self, d_net, real, fake):
        d_real = d_net(None, real, reuse=False)
        d_fake = d_net(None, fake, reuse=True)

        e = tf.random_uniform([tf.shape(fake)[0], 1, 1, 1], 0., 1., name='epsilon')
        fake_intp = real + e * (fake - real) 
        d_intp = d_net(None, fake_intp, reuse=True)
        gp = self._compute_gradient_penalty(d_intp, fake_intp)  
        return tf.reduce_mean(d_real), tf.reduce_mean(d_fake), gp 

    def loss(self):

        z = self._generate_noise_with_shape(self.real_clean)
        fake_clean = self.g_net(z, reuse=False)

        # Split into multiple frequency range to discriminate
        if len(self.d_net) > 1:
            split_fake = tf.split(fake_clean, len(self.d_net)-1, axis=2)
            split_real = tf.split(self.real_clean, len(self.d_net)-1, axis=2)
        # ===================
        # # For losses
        # ===================
        d_real_list = []
        d_fake_list = []
        d_gp_list = []
        self.d_losses = []
        self.g_losses = []

        # For full frequency range Discriminator
        dr, df, gp = self.w_loss(self.d_net[0], self.real_clean, fake_clean)
        d_real_list.append(dr)
        d_fake_list.append(df)
        d_gp_list.append(gp) 
        Wdist = dr - df
        self.d_losses.append(-Wdist + self.lamb_gp * gp)
        self.g_losses.append(-df)

        for i in range(len(self.d_net)-1):
            dr, df, gp = self.w_loss(self.d_net[i+1], split_real[i], split_fake[i])
            d_real_list.append(dr)
            d_fake_list.append(df)
            d_gp_list.append(gp)
            Wdist = dr - df
            self.d_losses.append(-Wdist + self.lamb_gp*gp)#+ 0.01*((tf.square(dr) + tf.square(df))/2))
            self.g_losses.append(-df)
             
        self.loss = dict()
        self.loss['l_G'] = 0
        for i in range(len(self.g_losses)):
            self.loss['l_G'] = self.loss['l_G'] + self.g_losses[i] 
        # ====================
        # Learning rate decay
        # ====================
        global_step = tf.Variable(0, trainable=False)
        decay_step = tf.maximum(0, (global_step - 90*1000))
        starter_learning_rate = self.lr
        learning_rate = tf.train.polynomial_decay(starter_learning_rate, decay_step, 90000, 0.0)
        # ===================
        # # For summaries
        # ===================
        sum_l_D = []
        for i in range(len(self.d_net)):
            sum_l_D.append(tf.summary.scalar('Wdist'+str(i), self.d_losses[i]))
            sum_l_D.append(tf.summary.scalar('GP'+str(i), d_gp_list[i]))
        sum_l_G = tf.summary.scalar('l_G', self.loss['l_G'])
        self.D_summs = sum_l_D
        self.G_summs = [sum_l_G]

        g_opt = None
        d_opt = [None] * len(self.d_net)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            for i in range(len(self.d_net)):    
                d_opt[i] = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, beta2=0.9)\
                    .minimize(self.d_losses[i], var_list=self.d_net[i].vars)
            g_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, beta2=0.9)\

        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement = True)
        config.gpu_options.allocator_type = 'BFC'
        self.sess = tf.Session(config=config)

        return d_opt, g_opt

    def _generate_noise_with_shape(self, x=None, batch_size=8):
        ''' iW-GAN used Gaussian noise '''
        # default mean=0.0, stddev=1.0
        with tf.name_scope('Z'):
            if x==None:
                z = tf.random_normal([batch_size, self.z_dim], name='z') 
            else:
                z = tf.random_normal([int(x.get_shape()[0]), self.z_dim], name='z') 
        return z

    def _compute_gradient_penalty(self, J, x, scope_name='GradientPenalty'):
        ''' Gradient Penalty
        Input:
            `J`: the loss
            `x`: shape = [b, c, h, w]
        '''
        with tf.name_scope(scope_name):
            grad = tf.gradients(J, x)[0]  # as the output is a list, [0] is needed
            grad_square = tf.square(grad)
            grad_squared_norm = tf.reduce_sum(grad_square, axis=[1, 2, 3])
            grad_norm = tf.sqrt(grad_squared_norm)
            # penalty = tf.square(tf.nn.relu(grad_norm - 1.)) # FIXME: experimental
            penalty = tf.square(grad_norm - 1.)
        return tf.reduce_mean(penalty)

    def train(self, mode="stage1", iters=65000):
        d_opt, g_opt = self.loss()

        self.sess.run(tf.global_variables_initializer())

        if tf.gfile.Exists(self.log_path+"D"):
            tf.gfile.DeleteRecursively(self.log_path+"D")
        tf.gfile.MkDir(self.log_path+"D")

        if tf.gfile.Exists(self.log_path+"G"):
            tf.gfile.DeleteRecursively(self.log_path+"G")
        tf.gfile.MkDir(self.log_path+"G")

        g_merged = tf.summary.merge(self.G_summs)
        d_merged = tf.summary.merge(self.D_summs)

        D_writer = tf.summary.FileWriter(self.log_path+"D", self.sess.graph)
        G_writer = tf.summary.FileWriter(self.log_path+"G", self.sess.graph)

        if mode == 'stage1':
            save_path = self.model_path
            print('Training:stage1')

        elif mode == 'stage2':
            save_path = self.model_path
            print('Training:stage2')

            with open(self.model_path + "checkpoint", 'r') as f:
                line = f.readline()
            latest_step = re.sub("[^0-9]", "", line)
            print(latest_step)
            with tf.device("/cpu:0"):
                saver = tf.train.Saver()
                saver.restore(self.sess, self.model_path + "model-" + latest_step)
        #-----------------------------------------------------------------#
        saver = tf.train.Saver(max_to_keep=100)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord,sess=self.sess)
        try:
            while not coord.should_stop():
                for i in range(iters):
                    if i%100==0:
                        fetch_list = d_opt + [g_opt, d_merged, g_merged]
                        fetched = self.sess.run(fetch_list)
                        # writer.add_summary(fetched[3], i)
                        # writer.add_summary(fetched[4], i)
                        # writer.add_summary(fetched[5], i)
                        # generated_n = self.sess.run([self.fake_noise])
                        # _, summary, loss_d = self.sess.run([d_opt, d_merged, self.loss['l_D']])    
                        D_writer.add_summary(fetched[-2], i)
                        # _, summary, loss_g = self.sess.run([g_opt, g_merged, self.loss['l_G']])
                        G_writer.add_summary(fetched[-1], i)
                        # _, summary, loss_g = self.sess.run([g_opt, g_merged, self.loss['l_G']])
                        # G_writer.add_summary(summary, i)
                        print("\rIter:{}".format(i))
                    else:
                        for _ in range(1):
                            _ = self.sess.run(d_opt)  
                        _ = self.sess.run([g_opt])

                    if i % 2000 == 1999:
                        saver.save(self.sess, save_path + 'model', global_step=i+1)
                    if i == iters-1:
                        coord.request_stop()

        except tf.errors.OutOfRangeError:
            print('Done training; epoch limit')
        finally:
            coord.request_stop()
        coord.join(threads)

        return
        
    def test(self, x_test, test_path, test_list):

        print('Testing:'+test_path)
        import scipy.io.wavfile as wav
        import librosa
        import scipy
        def filename2spec(filename, _max, _min):
            sr, y = wav.read(filename)
            if sr != 16000:
                raise ValueError('Sampling rate is expected to be 16kHz!')
            # For spectrum data
            if y.dtype!='float32':
                y = np.float32(y/32767.)
            spec, phase, mean, std = make_spectrum_phase(y, FRAMELENGTH, OVERLAP, _max, _min)
            slices = []
            for i in range(0, spec.shape[1]-FRAMELENGTH, OVERLAP):
                slices.append(spec[:,i:i+FRAMELENGTH])
            slices = np.array(slices).reshape((-1,1,257,FRAMELENGTH))
            slices = slices[:,:,1:,:]
            return slices

        def slice_signal(signal, window_size, overlap):
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

        def make_spectrum_phase(y, FRAMELENGTH, OVERLAP, _max=None, _min=None):
            D = librosa.stft(y,n_fft=512,hop_length=256,win_length=512,window=scipy.signal.hamming)
            Sxx = np.log10(abs(D)**2) 
            phase = np.exp(1j * np.angle(D))
            if _max.all() == None or _min.all() == None:
                mean = np.mean(Sxx, axis=1).reshape((257,1))
                std = np.std(Sxx, axis=1).reshape((257,1))+1e-12
                Sxx = (Sxx-mean)/std  
                return Sxx, phase, mean, std
            else:
                Sxx = 2 * (Sxx - _min)/(_max - _min) - 1       
                return Sxx, phase, None, None         

        def recons_spec_phase(Sxx_r, phase):
            Sxx_r = np.sqrt(10**Sxx_r)
            R = np.multiply(Sxx_r , phase)
            result = librosa.istft(R,
                             hop_length=256,
                             win_length=512,
                             window=scipy.signal.hamming)
            return result

        # From https://github.com/candlewill/Griffin_lim/blob/master/utils/audio.py
        def griffinlim(spectrogram, n_iter = 100, n_fft = 512, hop_length = 256):
            angles = np.exp(2j * np.pi * np.random.rand(*spectrogram.shape))

            for i in range(n_iter):
                full = np.abs(spectrogram).astype(np.complex) * angles
                inverse = librosa.istft(full, hop_length = hop_length, window = scipy.signal.hamming)
                rebuilt = librosa.stft(inverse, n_fft = n_fft, hop_length = hop_length, window = scipy.signal.hamming)
                angles = np.exp(1j * np.angle(rebuilt))

            full = np.abs(spectrogram).astype(np.complex) * angles
            inverse = librosa.istft(full, hop_length = hop_length, window = scipy.signal.hamming)

            return inverse

        _ = self.loss()

        if x_test.get_shape()[3] > 1:
            FRAMELENGTH = 64
            OVERLAP = 64
        else:
            FRAMELENGTH = 16384
            OVERLAP = 16384  

        predefined_z = np.random.normal(0.0, 1.0, [5,128])
        # Load model         
        with open(test_path + "checkpoint", 'r') as f:
            for line in f:
                latest_step = re.sub("[^0-9]", "", line)
                print("checkpoint = " + str(latest_step))
                with tf.device("/cpu:0"):
                    saver = tf.train.Saver()
                    saver.restore(self.sess, test_path + "model-" + latest_step)

                clist = [x[:-1] for x in open(test_list).readlines()]
                _max, _min = cal_minmax(clist)

                c_slices = filename2spec("p274_023_c.wav", _max, _min)
                n_slices = filename2spec("p274_023_n.wav", _max, _min)
                batch_size = c_slices.shape[0]

                _max = _max[1:, :]
                _min = _min[1:, :]
                # ====================================
                # Initial utterance from random noise
                # ===================================
                # z = self._generate_noise_with_shape(batch_size=5)
                z = tf.placeholder("float", [5, 128], name='random_noise')
                clean = self.g_net(z, reuse=True)
                generated_clean = self.sess.run(clean,feed_dict={z:predefined_z})
                full_spec = np.zeros((257,FRAMELENGTH*5))
                for i,c in enumerate(generated_clean):
                    print(c.shape)
                    c = (c+1)/2 * (_max - _min) + _min
                    print(np.max(c), np.min(c))
                    temp = np.sqrt(10**c)
                    # temp = np.clip(temp, -10, 10)
                    full_spec[1:,FRAMELENGTH*i:FRAMELENGTH*(i+1)] = temp

                out_wave = griffinlim(full_spec, n_iter = 50, n_fft = 512, hop_length = 256)
                check_dir(os.path.join(test_path, 'generated'))
                wav.write(os.path.join(test_path, 'generated')+"/fake_clean_"+str(latest_step)+".wav",
                        16000, np.int16(out_wave*32767.))
                # ==================================
                # Utterance after optimized on noisy
                # ==================================
                # print("Optimizing Input")
                # zv = tf.get_variable(name="noise",
                #                     shape=[batch_size, self.z_dim], 
                #                     dtype=tf.float32, 
                #                     initializer=tf.random_normal_initializer(0.0, 1.0))
                # # real_clean = tf.placeholder("float", [batch_size, 1, 256, FRAMELENGTH], name='real_clean')
                # clean = self.g_net(zv, reuse=True)
                # # d_fake = self.d_net[0](None, clean, reuse=True)
                # # d_real = self.d_net[0](None, real_clean, reuse=True)
                # mse_loss = tf.reduce_mean(tf.squared_difference(clean, x_test))
                # # w_dist = - tf.reduce_mean(d_fake)

                # opt = tf.train.GradientDescentOptimizer(learning_rate=10.0)\
                #     .minimize(mse_loss , var_list=zv)

                # init = tf.variables_initializer([zv])
                # self.sess.run(init)

                # best_loss = float("inf")
                # cnt = 0
                # for iters in range(1000):
                #     _, loss = self.sess.run([opt, mse_loss], 
                #                             feed_dict={x_test: c_slices})                    
                #     print(loss)

                #     if iters % 50 == 0:
                #         generated_clean = self.sess.run(clean)
                #         full_spec = np.zeros((257,FRAMELENGTH*batch_size))
                #         for i,c in enumerate(generated_clean):
                #             print(c.shape)
                #             c = (c+1)/2 * (_max - _min) + _min
                #             temp = np.sqrt(10**c)
                #             full_spec[1:,FRAMELENGTH*i:FRAMELENGTH*(i+1)] = temp

                #         out_wave = griffinlim(full_spec, n_iter = 100, n_fft = 512, hop_length = 256)
                #         wav.write("fake_enhancement.wav", 16000, np.int16(out_wave*32767.))

                #     # if best_loss <= loss:
                #     #     cnt += 1
                #     #     if cnt > 3 :
                #     #         break
                #     # else:
                #     #     best_loss = loss
                # break









