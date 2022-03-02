import numpy as np
import librosa
from matplotlib import pyplot as plt
from glob import glob
import os
from soundfile import read as audioread
from pathlib import Path
from librosa.filters import mel
from soundfile import write

class feature_extractor(object):
    def __init__(self):
        self.fs = 16000
        self.frm_length = 0.032
        self.nsampl = int(self.fs * self.frm_length)
        self.nfft = 512
        self.winlen = self.nsampl
        self.winsht = self.winlen
        self.nbin = int(self.nfft/2)+1
        self.freq = self.fs/self.nfft*np.arange(0, self.nbin)
        self.nchan = 1
        self.len_cut = 10*self.fs
        self.window = self.windowing(window='hanning', nx=self.nsampl)
        self.mel_basis = mel(sr=self.fs, n_fft=self.nfft)
        self.last = True

    def stft_mult_frame(self, aud):
        """
        :param aud: [chunk, nframe]
        :return: X: [nbins, nchan]
        """
        # Windowing
        aud = np.einsum('ij, i-> ij', aud, self.window)

        # Preallocation
        X = np.zeros((self.nbin, self.nframe), dtype=np.complex)
        # FFT
        for i in range(self.nchan):
            X[:, i] = np.fft.fft(a=aud[:, i], n=self.nfft)[:self.nbin]
        return X

    def stft_frame(self, aud_chunk):
        """
        :param aud_chunk:
        :return:
        """
        return

    def get_label(self, fn):
        npz_ = np.load(fn)
        label_sam = npz_['label']
        # if label_sam.shape[1] == 4:
        #     label_sam = label_sam[:, 0]
        return self.label_samp2frm(label_sam)

    def label_samp2frm(self, label_sample):
        """
        :param label_sample: [nsamples]
        :return:  [nframe, winlen]
        """

        label_frm = self.signal_to_frame(label_sample, last=self.last)
        label_frm = np.sum(np.abs(label_frm), axis=-1)
        label_frame = 1*(label_frm > int(self.nsampl*0.5))

        return label_frame

    def get_logmel(self, X):
        """
        :param X: [nbins, nframes]
        :return: [nmels, nframes]
        """
        # logmel
        log_mel = np.dot(self.mel_basis, np.abs(X) ** 2)
        log_mel = 10 * np.log10(np.maximum(1e-10, log_mel))

        return log_mel

    def wav2logmel(self, aud):
        # return aud
        aud=self.stft(aud)
        aud=self.get_logmel(aud)

        return aud

    def stft(self, aud):
        """
        :param aud: (nsamples,)
        :return: [nbins, nframe]
        """

        # signal to frame
        aud = self.signal_to_frame(aud, self.last) # [1, nsamples] -> [1, nframes, samples_per_frame]

        # Windowing
        aud = np.einsum('ij, j-> ij', aud, self.window) # [nframe, winlen]

        nframe = aud.shape[0]
        # Preallocation
        X = np.zeros((self.nbin, nframe), dtype=np.complex)

        # FFT
        for i in range(nframe):
            X[:, i] = np.fft.fft(a=aud[i, :], n=self.nfft)[:self.nbin]
        return X

    def audioread(self, fn):
        """
        :param path: audio path
        :return: audio signal
        """
        aud, fs = librosa.core.load(path=fn, sr=None, mono=False)
        return aud, fs

    def signal_to_frame(self, signal, last):
        """
        :param signal:
        :param last:
        :return:
        """
        # total utterance
        # signal_in: [time length]
        # signal_out: [feature, frames, frame_length]
        frame_length = self.winlen
        frame_step = self.winsht
        frames = signal.shape[0]

        # check if signal length is longer than frame length
        assert frames > frame_length
        assert frame_length >= frame_step

        if last == False:
            nframes = np.floor((frames - frame_length) / frame_step).astype(int) + 1
            frames_sig = np.zeros(shape=(nframes, frame_length), dtype=np.float32)
            for i in range(nframes):
                frames_sig[i, :] = signal[i * frame_step:frame_length + i * frame_step]
        else:
            nframes = np.floor((frames - frame_length) / frame_step).astype(int) + 2
            tot_frm = nframes * frame_length
            frames_sig = np.zeros(shape=(nframes, frame_length), dtype=np.float32)
            for i in range(nframes):
                if i == nframes - 1:
                    frames_sig[i, :] = np.concatenate(
                        (signal[i * frame_step:], np.zeros(shape=(tot_frm - frames))), axis=0)
                else:
                    frames_sig[i, :] = signal[i * frame_step:frame_length + i * frame_step]
        return frames_sig

    def windowing(self, window, nx):
        import scipy.signal.windows as windows
        from matplotlib import pyplot as plt
        hann = windows.get_window(window=window, Nx=nx)
        # plt.figure(1)
        # plt.plot(hann)
        # plt.show()
        return hann

    def _make_feature_target(self, fn):

        orig_path = '/'.join(fn.split('/')[:-1]) + '_feat/'
        Path(orig_path).mkdir(parents=True, exist_ok=True)
        npz_fn = fn.split('/')[-1]

        # data check
        new_fn = (orig_path+npz_fn)

        if os.path.exists(new_fn):
            pass
        else:
            GCC_pattern, X = self.get_GCCpattern(fn, False)
            log_mel = self.get_logmel(X)
            label = np.squeeze(self.get_label(fn))
            label = np.squeeze(label)

            np.savez(new_fn, feature=GCC_pattern.astype(np.float32), label=label, logmel=log_mel, X=X)

    def _make_feature_target_moving(self, fn):
        return


    def visualize_spectrogram(self, X_mult):
        """
        :param X_mult: [nchan, nframe, nbin]
        :return:
        """
        plt.figure(1, figsize=(8, 12), dpi=80, facecolor='w', edgecolor='k')
        for i in range(self.nchan):
            plt.subplot(2, 2, i + 1)
            plt.imshow(20 * np.log10(np.abs(X_mult[i, :, :])).T, aspect='auto')
            plt.gca().invert_yaxis()
        plt.show()

def main():

    ## audioread
    aud_path = '/media/jeonghwan/HDD2/0727/garo/rec1_barking/'

    # stereo to mono
    GCC_ext = feature_extractor()
    GCC_ext.stereo2mono(aud_path + 'rec1_bark-01.wav')


    # for i, wav_file in enumerate(wav_list):
    #     GCC_pattern = GCC_ext.get_GCCpattern(wav_file, False)[0]
    #     range_tdoa = np.linspace(-GCC_ext.range_tdoa * 1000, GCC_ext.range_tdoa * 1000, 5)
    #     plt.figure(1, figsize=(10, 9), dpi=80, facecolor='w', edgecolor='k')
    #
    #     GCC_pattern = GCC_pattern[:500, :, :]
    #
    #     for j in range(0, 6):
    #         plt.subplot(3, 2, j+1)
    #         plt.imshow(GCC_pattern[:, :, j].T, aspect='auto')
    #         plt.title('pair{}-{}'.format(GCC_ext.pairs[j][0]+1, GCC_ext.pairs[j][1]+1))
    #         plt.xlabel('frames')
    #         plt.ylabel('time delay (ms)')
    #         plt.gca().invert_yaxis()
    #         plt.axis([0, GCC_pattern.shape[0], 0, GCC_pattern.shape[1]]) # axis -> ([xmin, xmax, ymin, ymax])
    #         plt.yticks(np.linspace(0, GCC_ext.ngrid, 5), [tdoa for tdoa in range_tdoa]) # ytick -> (real values, labels)
    #         #plt.jet()
    #     plt.show()

# def calculate_SRP():
#
#     # delay-and-sum beamformer
#
#     tde =
#
#     # MVDR beamformer
#
#
#
#     # MPDR beamformer
#
#     # MUSIC spectrum
#
#     #
#
    return

def calculate_SNR():
    """
    :param signal: [nsamples, ]
    :param noise: [nsamples, ]
    :return:
    """

    feat_ext = feature_extractor()

    signal, fs = feat_ext.audioread('/media/jeonghwan/HDD2/0706/rec/moving_test_rec_ample_1.wav')
    noise, fs = feat_ext.audioread('/media/jeonghwan/HDD2/0706/moving/moving_noise_1.wav')

    for i, t in enumerate(np.linspace(0, 70, 8)):
        init_t = int(t)
        signal_temp = signal[init_t*fs:(init_t+10)*fs]
        noise_temp  = noise[init_t*fs:(init_t+10)*fs]

        #
        feat_ext.audiowrite('sig{}.wav'.format(i), signal_temp, fs)
        feat_ext.audiowrite('noi{}.wav'.format(i), noise_temp, fs)

        sig_frm = feat_ext.signal_to_frame(signal_temp[np.newaxis, :], frame_length=feat_ext.winlen, frame_step=feat_ext.winsht, last=False)
        noi_frm = feat_ext.signal_to_frame(noise_temp[np.newaxis, :], frame_length=feat_ext.winlen, frame_step=feat_ext.winsht, last=False)

        nfrm = np.minimum(sig_frm.shape[1], noi_frm.shape[1])

        sig_pwr = np.zeros((nfrm, 1))
        noi_pwr = np.zeros((nfrm, 1))

        for i in range(nfrm):
            sig_pwr[i, 0] = np.sum(sig_frm[0, i, :] ** 2)
            noi_pwr[i, 0] = np.sum(noi_frm[0, i, :] ** 2)

       #  plt.figure(1, figsize=(12, 6), dpi=200, facecolor='w', edgecolor='k')
       #  plt.subplot(3, 2, 3) # signal energy
       #  plt.plot(sig_pwr)
       #  plt.title('signal power')
       #  plt.ylim([0, 0.15])
       #  plt.xticks([], [])
       #  plt.subplot(3, 2, 4) # noise energy
       #  plt.plot(noi_pwr)
       #  plt.ylim([0, 0.15])
       #  plt.title('noise power')
       #  plt.xticks([], [])
       #  plt.subplot(3, 2, 5) # SNR
       #  plt.plot(10*np.log10(sig_pwr/noi_pwr))
       #  #plt.title('signal-to-noise ratio (SNR)')
       #  plt.ylim([-20, 20])
       #  plt.ylabel('dB')
       #  plt.xlabel('frames')
       #
       # # print(np.mean(10*np.log10()))
       #  print(np.mean(10*np.log10(sig_pwr/noi_pwr)))
       #  #plt.show()
       #
       #  plt.subplot(3, 2, 1)  # signal energy
       #  plt.plot(signal_temp)
       #  plt.title('sound events')
       #  plt.ylim([-0.05, 0.05])
       #  # plt.xticks([0, 10000, 20000, 30000, 40000, 50000], [])
       #  plt.xticks([], [])
       #
       #  plt.subplot(3, 2, 2)  # noise energy
       #  plt.plot(noise_temp)
       #  plt.title('ego-noise')
       #  plt.ylim([-0.05, 0.05])
       #  plt.xticks([], [])
       #  #plt.subplot(3, 1, 3)  # SNR
       #  #plt.plot(10 * np.log10(sig_pwr / noi_pwr))
       #  #plt.xlabel('samples')
       #  # plt.show()
       #
       #  plt.savefig('test{}.png'.format(t))
       #  plt.close()
    # SNR

def calculate_SNR2():
    """
    :param signal: [nsamples, ]
    :param noise: [nsamples, ]
    :return:
    """
    feat_ext = feature_extractor()

    signal, fs = feat_ext.audioread('/media/jeonghwan/HDD2/0706/rec_moving/moving_test_rec_ample_moving_1.wav')
    #noise, fs = feat_ext.audioread('/media/jeonghwan/HDD2/0706/moving/moving_noise_1.wav')

    for i, t in enumerate(np.linspace(0, 70, 8)):
        init_t = int(t)
        signal_temp = signal[init_t*fs:(init_t+10)*fs]
        #noise_temp  = noise[init_t*fs:(init_t+10)*fs]

        #
        #feat_ext.audiowrite('noisy{}.wav'.format(i), signal_temp, fs)
        #feat_ext.audiowrite('noi{}.wav'.format(i), noise_temp, fs)


        sig_frm = feat_ext.signal_to_frame(signal_temp[np.newaxis, :], frame_length=feat_ext.winlen, frame_step=feat_ext.winsht, last=False)
        #noi_frm = feat_ext.signal_to_frame(noise_temp[np.newaxis, :], frame_length=feat_ext.winlen, frame_step=feat_ext.winsht, last=False)

        #nfrm = np.minimum(sig_frm.shape[1], noi_frm.shape[1])

        # sig_pwr = np.zeros((nfrm, 1))
        # noi_pwr = np.zeros((nfrm, 1))
        #
        # for i in range(nfrm):
        #     sig_pwr[i, 0] = np.sum(sig_frm[0, i, :] ** 2)
        #     noi_pwr[i, 0] = np.sum(noi_frm[0, i, :] ** 2)

        plt.figure(1, figsize=(12, 6), dpi=200, facecolor='w', edgecolor='k')
        # plt.subplot(3, 2, 3) # signal energy
        # plt.plot(sig_pwr)
        # plt.title('signal power')
        # plt.ylim([0, 0.15])
        # plt.xticks([], [])
        # plt.subplot(3, 2, 4) # noise energy
        # plt.plot(noi_pwr)
        # plt.ylim([0, 0.15])
        # plt.title('noise power')
        # plt.xticks([], [])
        # plt.subplot(3, 2, 5) # SNR
        # plt.plot(10*np.log10(sig_pwr/noi_pwr))
        # #plt.title('signal-to-noise ratio (SNR)')
        # plt.ylim([-20, 20])
        # plt.ylabel('dB')
        # plt.xlabel('frames')

       # # print(np.mean(10*np.log10()))
       #  print(np.mean(10*np.log10(sig_pwr/noi_pwr)))
       #  #plt.show()
       #
        plt.subplot(3, 2, 1)  # signal energy
        plt.plot(signal_temp)
        plt.title('sound events')
        plt.ylim([-0.05, 0.05])
        # plt.xticks([0, 10000, 20000, 30000, 40000, 50000], [])
        plt.xticks([], [])

       #  plt.subplot(3, 2, 2)  # noise energy
       #  plt.plot(noise_temp)
       #  plt.title('ego-noise')
       #  plt.ylim([-0.05, 0.05])
       #  plt.xticks([], [])
       #  #plt.subplot(3, 1, 3)  # SNR
       #  #plt.plot(10 * np.log10(sig_pwr / noi_pwr))
       #  #plt.xlabel('samples')
       #  # plt.show()

        plt.savefig('test_noisy{}.png'.format(t))
        plt.close()

    # SNR

def save_mat(path):

    #path = '/media/jeonghwan/HDD2/0727/sero/wn_sero/'
    from scipy.io import savemat

    npy_list = glob(path + '*.npy')
    npy = np.load(npy_list[0], allow_pickle=True)
    print('h')
    # dp = data_preparator()
    # pos_list, t_list = dp._read_pozyx_from_npy(npy)

    xy_list = npy[0::2]
    time_stamps = npy[1::2]
    t_list = [float(t) for t in time_stamps]
    x_list = [xy.x for xy in xy_list]
    y_list = [xy.y for xy in xy_list]

    # save mat
    savemat(npy_list[0].replace('.npy', '.mat'), {'x': x_list, 'y': y_list, 't': t_list})


if __name__=='__main__':
    #save_mat('/media/jeonghwan/HDD2/0727/garo/rec1_barking/')
    # feat_ext = feature_extractor()
    # feat_ext.label_samp2frm()
    #
    # calculate_SNR2()
    main()

    # feat_cls = feature_extractor()
    # feat_cls.stereo2mono('/media/jeonghwan/HDD2/0722/rec2-01.wav')