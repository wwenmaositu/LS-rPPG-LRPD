import numpy as np
import os
import h5py
from torch.utils.data import Dataset
from PIL import Image
import random
# from scipy.fft import fft
# from scipy import signal
# from scipy.signal import butter, filtfilt
# from sklearn.model_selection import KFold

def UBFC_LU_split():
    # split UBFC dataset into training and testing parts
    # the function returns the file paths for the training set and test set.
    # TODO: if you want to train on another dataset, you should define new train-test split function.
    
    h5_dir = ''
    train_list = []
    val_list = []

    val_subject = []

    for subject in range(1,50):
        if os.path.isfile(h5_dir+'/%d.h5'%(subject)):
            if subject in val_subject:
                val_list.append(h5_dir+'/%d.h5'%(subject))
            else:
                train_list.append(h5_dir+'/%d.h5'%(subject))

    return train_list, val_list    

def PURE_split():

    h5_dir = ''
    train_list = []
    val_list = []

    val_subject = []

    for subject in range(1,11):
        for sess in [1,2,3,4,5,6]:
            if os.path.isfile(h5_dir+'/%d/%02d.h5'%(subject, sess)):
            # if os.path.isfile(os.path.join(h5_dir, str(subject), '%02d.h5' % (sess))):
                if subject in val_subject:
                    val_list.append(h5_dir+'/%d/%02d.h5'%(subject, sess))
                else:
                    train_list.append(h5_dir+'/%d/%02d.h5'%(subject, sess))
    return train_list, val_list




def BUAA_split():

    h5_dir = ''
    train_list = []
    val_list = []

    val_subject = []
    # print("textdatasets are:",val_subject)

    for subject in range(1,15):
        if os.path.isfile(h5_dir+'/%02d.h5'%(subject)):
            if subject in val_subject:
                val_list.append(h5_dir+'/%02d.h5'%(subject))
            else:
                train_list.append(h5_dir+'/%02d.h5'%(subject))

    return train_list, val_list


# def VIPLHR_split():
#     h5_dir = ''
#     train_list = []
#     val_list = []
#
#     val_subject = [f'p{s}' for s in range(86,108)]
#     for i in range(1, 108):
#         subject = f'p{i}'
#         for j in range(1,10):
#             sess = f'v{j}'
#             for k in(1,4):
#                 access = f'source{k}'
#                 if os.path.isfile(f"{h5_dir}/{subject}/{sess}/{access}.h5"):
#                     if subject in val_subject:
#                         val_list.append(f"{h5_dir}/{subject}/{sess}/{access}.h5")
#                     else:
#                         train_list.append(f"{h5_dir}/{subject}/{sess}/{access}.h5")
#
#     return train_list,val_list


def MRNIRP_split():

    h5_dir = ''
    train_list = []
    val_list = []

    val_subject = []

    for subject in range(1, 16):
        if os.path.isfile(h5_dir + '/%d.h5' % (subject)):
            if subject in val_subject:
                val_list.append(h5_dir + '/%d.h5' % (subject))
            else:
                train_list.append(h5_dir + '/%d.h5' % (subject))

    return train_list, val_list


def LRPD_split():
    h5_dir = ""  # 这里替换成你实际存放数据的文件夹路径
    train_list = []
    test_list = []

    val_subject = []

    for subject in range(1, 43):
        subject_name = f'P{subject}'
        subject_path = os.path.join(h5_dir, subject_name)
        if os.path.isdir(subject_path):
            lux_folder_path = os.path.join(subject_path, ' ')
            if os.path.isdir(lux_folder_path):
                for file in [' ']:
                    file_path = os.path.join(lux_folder_path, file)
                    if os.path.isfile(file_path):
                        if subject in val_subject:
                            test_list.append(file_path)
                        else:
                            train_list.append(file_path)

    return train_list, test_list


class H5Dataset(Dataset):

    def __init__(self, train_list, T):
        self.train_list = train_list # list of .h5 file paths for training
        self.T = T # video clip length

    def __len__(self):
        return len(self.train_list)

    def horizontal_flip(self, img_seq):
        # 随机决定是否进行水平翻转
        if np.random.rand() < 0.5:
            # 对视频序列中的每一帧图像进行翻转
            img_seq = img_seq[:, :, ::-1, :]
        return img_seq

    def rotate(self, img_seq, angle):
        # 指定旋转角度
        if np.random.rand() < 0.5:
            # 对每帧图像进行旋转
            for i in range(img_seq.shape[0]):
                img = Image.fromarray(img_seq[i].astype('uint8'))  # 将 numpy 数组转换为 PIL 图像
                img = img.rotate(angle)  # 旋转图像
                img_seq[i] = np.array(img)  # 将 PIL 图像转换回 numpy 数组
        return img_seq



    def __getitem__(self, idx):
        with h5py.File(self.train_list[idx], 'r') as f:
            # img_length = f['imgs'].shape[0]
            img_length = np.min([f['imgs'].shape[0]])

            if img_length <= self.T:
                raise ValueError(f"Video at {self.train_list[idx]} is too short for the clip length T={self.T}")
            choice_range = img_length - self.T
            # 如果可选择的范围大于1，则进行随机抽样
            if choice_range > 1:
                idx_start = np.random.choice(choice_range)
            else:  # 如果恰好等于1，直接选择0
                idx_start = 0

            # idx_start = np.random.choice(img_length-self.T)

            idx_end = idx_start+self.T

            img_seq = f['imgs'][idx_start:idx_end]
            # img_seq = self.horizontal_flip(img_seq)
            # img_seq = self.rotate(img_seq,15)
            img_seq = np.transpose(img_seq, (3, 0, 1, 2)).astype('float32')
        return img_seq


class H5Dataset_(Dataset):

    def __init__(self, val_list, T):
        self.val_list = val_list # list of .h5 file paths for training
        self.T = T # video clip length

    def __len__(self):
        return len(self.val_list)

    def __getitem__(self, idx):
        with h5py.File(self.val_list[idx], 'r') as f:
            # img_length = f['imgs'].shape[0]
            img_length = np.min([f['imgs'].shape[0], f['bvp'].shape[0]])

            idx_start = np.random.choice(img_length-self.T)

            idx_end = idx_start+self.T

            bvp = f['bvp'][idx_start:idx_end].astype('float32')

            img_seq = f['imgs'][idx_start:idx_end]
            img_seq = np.transpose(img_seq, (3, 0, 1, 2)).astype('float32')
        return img_seq,bvp



