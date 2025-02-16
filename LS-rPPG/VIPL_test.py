import numpy as np
import h5py
import torch
# from MYModel import *
from LSPhysModel import LSPhysNet
from utils_data import *
from utils_sig import *
from sacred import Experiment
from sacred.observers import FileStorageObserver
import json
import os

from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

np.set_printoptions(threshold=np.inf)
ex = Experiment('model_pred', save_git_info=False)

# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'


@ex.config
def my_config():
    # e = 0 # the model checkpoint at epoch e
    train_exp_num = 1 # the training experiment number
    train_exp_dir = '' % train_exp_num  # training experiment directory
    time_interval = 30  # get rppg for 30s video clips, too long clips might cause out of memory

    ex.observers.append(FileStorageObserver(train_exp_dir))

    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        num_gpus = torch.cuda.device_count()

    else:
        device = torch.device('cpu')
        num_gpus = 0


@ex.automain
def my_main(_run, train_exp_dir, device, time_interval):
    # load test file paths
    test_list = list(np.load(train_exp_dir + '/test_list.npy'))
    # pred_exp_dir = train_exp_dir + '/%d' % (int(_run._id))  # prediction experiment directory

    with open(train_exp_dir + '/config.json') as f:
        config_train = json.load(f)

    model = LSPhysNet(config_train['S'], config_train['in_ch']).to(device).eval()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)

    # model.load_state_dict(
    #     torch.load(train_exp_dir + '/epoch%d.pt' % (e), map_location=device))  # load weights to the model

    @torch.no_grad()
    def dl_model(imgs_clip):
        # model inference
        img_batch = imgs_clip
        img_batch = img_batch.transpose((3, 0, 1, 2))
        img_batch = img_batch[np.newaxis].astype('float32')
        img_batch = torch.tensor(img_batch).to(device)

        rppg = model(img_batch)[:, -1, :]
        rppg = rppg[0].detach().cpu().numpy()
        return rppg

    for e in range(30):  # Loop over epochs from 0 to 29
        model.load_state_dict(
            torch.load(train_exp_dir + '/epoch%d.pt' % e, map_location=device))  # load weights to the model
        pred_exp_dir = train_exp_dir + '/%d' % (int(_run._id) + e)  # prediction experiment directory

        if not os.path.exists(pred_exp_dir):
            os.makedirs(pred_exp_dir)

        # # Check and print learnable_scale and channel_weights
        # learnable_scale = model.module.learnable_scale if isinstance(model,
        #                                                              torch.nn.DataParallel) else model.learnable_scale
        # channel_weights = model.module.channel_weights if isinstance(model,
        #                                                              torch.nn.DataParallel) else model.channel_weights
        # print(f"Epoch {e}: learnable_scale = {learnable_scale.detach().cpu().numpy()}")
        # print(f"Epoch {e}: channel_weights = {channel_weights.detach().cpu().numpy()}")

        rppg_hr = []
        bvp_hr = []
        for h5_path in test_list:
            h5_path = str(h5_path)

            with h5py.File(h5_path, 'r') as f:
                imgs = f['imgs']
                # bvp = f['hr'][:]
                bvp = f['bvp'][:]
                # bvppeak = f['bvp_peak']
                fs = config_train['fs']

                duration = np.min([imgs.shape[0]]) / fs
                num_blocks = int(duration // time_interval)

                rppg_list = []
                bvp_list = []
                # bvppeak_list = []

                for b in range(num_blocks):
                    with torch.no_grad():
                        img_clip = imgs[b * time_interval * fs:(b + 1) * time_interval * fs]
                        rppg_clip = dl_model(imgs[b * time_interval * fs:(b + 1) * time_interval * fs])
                    rppg_list.append(rppg_clip)

                    bvp_list.append(bvp[b * time_interval * fs:(b + 1) * time_interval * fs])
                    # bvppeak_list.append(bvppeak[b*time_interval*fs:(b+1)*time_interval*fs])

                rppg_list = np.array(rppg_list)
                bvp_list = np.array(bvp_list)
                rppg0 = butter_bandpass(rppg_list, lowcut=0.6, highcut=4, fs=30)
                hr0, psd_y0, psd_x0 = hr_fft(rppg0, fs=30)

                bvp0 = butter_bandpass(bvp_list, lowcut=0.6, highcut=4, fs=30)
                hr1, psd_y1, psd_x1 = hr_fft(bvp0, fs=30)
                # hr1 = np.mean(bvp)
                rppg_hr.append(hr0)
                bvp_hr.append(hr1)

                # bvppeak_list = np.array(bvppeak_list)
                # results = {'rppg_list': rppg_list, 'bvp_list': bvp_list, 'bvppeak_list':bvppeak_list}
                results = {'rppg_list': rppg_list, 'bvp_list': bvp_list}
                np.save(pred_exp_dir + '/' + h5_path.split('/')[-1][:-3],results)