from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import json
import glob
import random
import collections
import math, time, sys, os

from data_utils import *
from ops import *
from model import *
from dnn_model import *
from cwgan import *

use_waveform = False
batch_size = 8
learning_rate = 1e-4
iters = 180000
FRAMELENGTH = 64
OVERLAP = 64

mode = sys.argv[1] # stage1, stage2, test
log_path = 'stage1_log/20180103-mhint'
model_path = 'stage1_model/model_20180103-segan/'
model_path2 = 'dummy/'
test_path = model_path # switch between stage1 and stage2
test_list = "/mnt/gv0/user_sylar/TMHINT/tsnoisylist"
record_path = "/mnt/hd-01/user_sylar/MHINTSYPD_100NS"
record_name = "/data_spec_clean_minmax.tfrecord"
# record_path = "/mnt/gv0/user_sylar/segan_data"
# record_name = "/p274_spec_clean_minmax.tfrecord"

if use_waveform:
    E=Enhancer((1,2,3),(1,2,3))
    NG=Noise_generator((1,2,3),(1,2,3))
    Dn=Discriminator()
else:
    G=Noise_generator((1,256,64),(1,256,64))
    D=glob_Discriminator(name='D_full')
    D1=Discriminator(name='D1')
    D2=Discriminator(name='D2')
    D3=Discriminator(name='D3')
    D4=Discriminator(name='D4')

check_dir(log_path)
check_dir(model_path)
check_dir(model_path2)

with tf.device('cpu'):
    reader = dataPreprocessor(record_path, record_name, use_waveform=use_waveform)
    clean, noisy = reader.read_and_decode(batch_size=batch_size,num_threads=32)
#with tf.device('gpu'):
gan = GradientPenaltyWGAN(G,[D],noisy,clean,log_path,model_path,use_waveform,lr=learning_rate)

if mode=='test':
    if use_waveform:
        x_test = tf.placeholder("float", [None, 1, FRAMELENGTH, 1], name='test_noisy')
    else:
        x_test = tf.placeholder("float", [None, 1, 256, FRAMELENGTH], name='test_noisy')
    gan.test(x_test, test_path, test_list)
else:
    gan.train(mode, iters)

# ======================
# Creating tfrecord
# ======================
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

# import tensorflow as tf
# import numpy as np
# import argparse
# import os
# import json
# import glob
# import random
# import collections
# import math
# import time
# import sys

# from data_utils import *
# from ops import *
# from model import *
# from dnn_model import *
# from cwgan import *

# use_waveform = False
# record_path = "/mnt/gv0/user_sylar/segan_data"
# record_name = "/p274_spec_clean_minmax.tfrecord"
# reader = dataPreprocessor(record_path, record_name, 
#                         noisy_filelist='/mnt/gv0/user_sylar/segan_data/p274_clean_list',
#                         clean_filelist='/mnt/gv0/user_sylar/segan_data/p274_clean_list',
#                         use_waveform=use_waveform)
# reader.write_tfrecord()