import numpy as np
from scipy.fft import fft
from scipy import signal
from scipy.signal import butter, filtfilt

def butter_bandpass(sig, lowcut, highcut, fs, order=2):
    # butterworth bandpass filter
    
    sig = np.reshape(sig, -1)
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    
    y = filtfilt(b, a, sig)
    return y

def butter_bandpass_batch(sig_list, lowcut, highcut, fs, order=2):
    # butterworth bandpass filter (batch version)
    # signals are in the sig_list

    y_list = []
    
    for sig in sig_list:
        sig = np.reshape(sig, -1)
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, sig)
        y_list.append(y)
    return np.array(y_list)

def hr_fft(sig, fs, harmonics_removal=True):
    # get heart rate by FFT
    # return both heart rate and PSD

    sig = sig.reshape(-1)
    sig = sig * signal.windows.hann(sig.shape[0])
    sig_f = np.abs(fft(sig))
    low_idx = np.round(0.6 / fs * sig.shape[0]).astype('int')
    high_idx = np.round(4 / fs * sig.shape[0]).astype('int')
    sig_f_original = sig_f.copy()
    
    sig_f[:low_idx] = 0
    sig_f[high_idx:] = 0

    peak_idx, _ = signal.find_peaks(sig_f)
    sort_idx = np.argsort(sig_f[peak_idx])
    sort_idx = sort_idx[::-1]

    peak_idx1 = peak_idx[sort_idx[0]]
    peak_idx2 = peak_idx[sort_idx[1]]

    f_hr1 = peak_idx1 / sig.shape[0] * fs
    hr1 = f_hr1 * 60

    f_hr2 = peak_idx2 / sig.shape[0] * fs
    hr2 = f_hr2 * 60
    if harmonics_removal:
        if np.abs(hr1-2*hr2)<10:
            hr = hr2
        else:
            hr = hr1
    else:
        hr = hr1

    x_hr = np.arange(len(sig))/len(sig)*fs*60
    return hr, sig_f_original, x_hr


def hr_fft_min(sig, fs, harmonics_removal=True):
    sig = sig.reshape(-1)
    sig = sig * signal.windows.hann(sig.shape[0])
    sig_f = np.abs(fft(sig))

    # 频率范围限制
    low_idx = int(0.6 / fs * len(sig))
    high_idx = int(4 / fs * len(sig))
    sig_f[:low_idx] = 0
    sig_f[high_idx:] = 0

    # 带最小高度的峰值检测
    min_height = np.percentile(sig_f, 75)  # 只考虑前25%的强度
    peak_idx, _ = signal.find_peaks(sig_f, height=min_height)

    # 处理无峰值情况
    if len(peak_idx) == 0:
        return 0.0, sig_f, np.arange(len(sig)) / len(sig) * fs * 60

    # 获取主峰
    main_peak = peak_idx[np.argmax(sig_f[peak_idx])]
    hr1 = main_peak / len(sig) * fs * 60

    # 处理单一峰值情况
    if len(peak_idx) < 2:
        return hr1, sig_f, np.arange(len(sig)) / len(sig) * fs * 60

    # 获取次峰
    secondary_peaks = np.delete(peak_idx, np.argmax(sig_f[peak_idx]))
    sec_peak = secondary_peaks[np.argmax(sig_f[secondary_peaks])]
    hr2 = sec_peak / len(sig) * fs * 60

    # 谐波修正
    if harmonics_removal:
        if abs(hr1 - 2 * hr2) < 10:  # 主峰是次峰的二次谐波
            return hr2, sig_f, np.arange(len(sig)) / len(sig) * fs * 60
    return hr1, sig_f, np.arange(len(sig)) / len(sig) * fs * 60


def hr_fft_batch(sig_list, fs, harmonics_removal=True):
    # get heart rate by FFT (batch version)
    # return both heart rate and PSD

    hr_list = []
    for sig in sig_list:
        sig = sig.reshape(-1)
        sig = sig * signal.windows.hann(sig.shape[0])
        sig_f = np.abs(fft(sig))
        low_idx = np.round(0.6 / fs * sig.shape[0]).astype('int')
        high_idx = np.round(4 / fs * sig.shape[0]).astype('int')
        sig_f_original = sig_f.copy()
        
        sig_f[:low_idx] = 0
        sig_f[high_idx:] = 0

        peak_idx, _ = signal.find_peaks(sig_f)
        sort_idx = np.argsort(sig_f[peak_idx])
        sort_idx = sort_idx[::-1]

        peak_idx1 = peak_idx[sort_idx[0]]
        peak_idx2 = peak_idx[sort_idx[1]]

        f_hr1 = peak_idx1 / sig.shape[0] * fs
        hr1 = f_hr1 * 60

        f_hr2 = peak_idx2 / sig.shape[0] * fs
        hr2 = f_hr2 * 60
        if harmonics_removal:
            if np.abs(hr1-2*hr2)<10:
                hr = hr2
            else:
                hr = hr1
        else:
            hr = hr1

        # x_hr = np.arange(len(sig))/len(sig)*fs*60
        hr_list.append(hr)
    return np.array(hr_list)

def normalize(x):
    return (x-x.mean())/x.std()


def SNR_get(waveform, gt_hr, fs, filtered=False):
    waveform = np.reshape(waveform, -1)
    if filtered:
        waveform = butter_bandpass(waveform, 0.6, 4, 60)
    N = waveform.shape[0]

    #     low_idx = np.round(0.6 / fs * waveform.shape[0]).astype('int')
    #     high_idx = np.round(4 / fs * waveform.shape[0]).astype('int')

    #     waveform[:low_idx] = 0
    #     waveform[high_idx:] = 0

    bin1 = round(5 / 60 / fs * N)
    bin2 = round(10 / 60 / fs * N)

    f1 = gt_hr / 60
    f2 = f1 * 2

    bc1 = round(f1 * N / fs)
    bc2 = round(f2 * N / fs)

    window = signal.windows.hann(N)
    win_waveform = waveform * window
    waveform_f = np.abs(fft(win_waveform)) ** 2

    total_power = np.sum(waveform_f)
    signal_power1 = 2 * np.sum(waveform_f[bc1 - bin1:bc1 + bin1])
    signal_power2 = 2 * np.sum(waveform_f[bc2 - bin2:bc2 + bin2])

    signal_power = signal_power1 + signal_power2
    noise_power = total_power - signal_power

    snr = 10 * np.log10(signal_power / noise_power)
    return snr