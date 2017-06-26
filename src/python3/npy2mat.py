# -*- coding:utf-8 -*-

import numpy as np 
import scipy.io as sio


def transform(round, flag, len_stips):
    print('../data/common/cline_r%d_f%d_s%d.npy'%(round,flag,len_stips))
    cline = np.load('../data/common/cline_r%d_f%d_s%d.npy'%(round,flag,len_stips))

    print('../data/common/label_r%d_f%d_s%d.npy'%(round,flag,len_stips))
    label = np.load('../data/common/label_r%d_f%d_s%d.npy'%(round,flag,len_stips))

    print('../data/common/stips_r%d_f%d_s%d.npy'%(round,flag,len_stips))
    stips = np.load('../data/common/stips_r%d_f%d_s%d.npy'%(round,flag,len_stips))

    print('../data/common/cline_label_stips_r%d_f%d_s%d'%(round,flag,len_stips))
    sio.savemat('../data/common/cline_label_stips_r%d_f%d_s%d'%(round,flag,len_stips),{'cline':cline,'label':label,'stips':stips})

transform(1,2,2227730)
