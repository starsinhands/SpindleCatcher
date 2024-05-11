"""
- Author: Federico Lionetti
- Course: Biosignal Processing
- Faculty & Degree: UAS TECHNIKUM WIEN - Master's Degree Program: Medical Engineering & eHealth
- Date: 02 Nov. 2021


Hereby the author  present you a "Sleep Spindle" Detector that from raw EEGs files (the file format is .edf) finds,
highlights and then visually presents the sleep spindles that it can detect.
To do this, the code uses 4 parameters: 1) Absolute Sigma Power; 2) Relative Sigma Power; 3) Moving Correlation;
4) Moving Root-Mean-Square.

In creating this code, the author relied on public codes of others sleep spindle detectors (i.e., YASA and Wonambi)
Please refer to these links for checking out the above-mentioned sources:
- https://raphaelvallat.com/yasa/build/html/index.html#
- https://wonambi-python.github.io/introduction.html

#######################################################################################################################

How the function 'detect_spindles' works
    - Input:
            eeg: Contains the input EEG signal as column vector
            fs: Sampling rate of the EEG signal
    - Output:
            spindles: Contains an n-by-2 matrix where each row corresponds to a detected sleep spindle and the first
            column contains the start time of spindle in seconds from the start of the recording and the second column
            contains the duration of the spindle in seconds.
"""

import mne
import numpy as np
import pickle
import time
import json
import matplotlib.pyplot as plt
import pandas as pd
from mne.filter import filter_data
from pathlib import Path
from scipy import signal
from scipy import stats
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import interp1d


class tool_precitor:

    def detect_spindles(self,eeg, fs):
            start_time = time.perf_counter()

            ABSOLUTE_SIGMA_POWER_THRESHOLD = 1.25
            RELATIVE_SIGMA_POWER_THRESHOLD = 0.20
            MOVING_CORRELATION_THRESHOLD = 0.69
            MOVING_RMS_THRESHOLD = 1.5

            THRESHOLD_FOR_GOOD_STUFF = 2.67
            DISTANCE_BETWEEN_SPINDLES = 500
            MAX_TIME_FOR_SPINDLES = 2
            MIN_TIME_FOR_SPINDLES = 0.5

            file_name = eeg

            sample_freq = fs

            raw_data = np.load(eeg,allow_pickle=True)

            timesA = np.arange(raw_data.size) / sample_freq

            freq_board = [1, 30]
            data_broad = filter_data(raw_data, sample_freq, freq_board[0], freq_board[1], method='fir', verbose=0)

            dat_sigma_w = data_broad.copy()
            N = 20
            nyquist = sample_freq / 2
            Wn = 11 / nyquist
            sos = signal.iirfilter(N, Wn, btype='Highpass', output='sos')
            dat_sigma_w = signal.sosfiltfilt(sos, dat_sigma_w)
            Wn = 16 / nyquist
            sos = signal.iirfilter(N, Wn, btype='lowpass', output='sos')
            dat_sigma_w = signal.sosfiltfilt(sos, dat_sigma_w)

            dur = 0.3
            halfdur = dur / 2
            total_dur = len(dat_sigma_w) / sample_freq
            last = len(dat_sigma_w) - 1
            step = 0.1

            len_out = int(len(dat_sigma_w) / (step * sample_freq))
            out = np.zeros(len_out)
            tt = np.zeros(len_out)

            for i, j in enumerate(np.arange(0, total_dur, step)):
                beg = max(0, int((j - halfdur) * sample_freq))
                end = min(last, int((j + halfdur) * sample_freq))
                tt[i] = (np.column_stack((beg, end)).mean(1) / sample_freq)
                out[i] = np.mean(np.square(dat_sigma_w[beg:end]))

            dat_det_w = out
            dat_det_w[dat_det_w <= 0] = 0.000000001
            abs_sig_pow = np.log10(dat_det_w)

            interop = interp1d(tt, abs_sig_pow, kind='cubic', bounds_error=False,
                               fill_value=0, assume_sorted=True)

            tt = np.arange(dat_sigma_w.size) / sample_freq
            abs_sig_pow = interop(tt)

            text = 'spindles'
            spindles_counter_method0 = {}
            name = 0
            for item in abs_sig_pow:
                if item >= ABSOLUTE_SIGMA_POWER_THRESHOLD:
                    spindles_counter_method0['item' + str(name)] = [item]
                else:
                    name += 1
            if len(spindles_counter_method0) == 1:
                text = 'spindle'

            f, t, SXX = signal.stft(data_broad, sample_freq, nperseg=(2 * sample_freq),
                                    noverlap=((2 * sample_freq) - (0.2 * sample_freq)))

            idx_band = np.logical_and(f >= freq_board[0], f <= freq_board[1])
            f = f[idx_band]
            SXX = SXX[idx_band, :]
            SXX = np.square(np.abs(SXX))
            PIPPO = RectBivariateSpline(f, t, SXX)
            t = np.arange(data_broad.size) / sample_freq
            SXX = PIPPO(f, t)
            sum_pow = SXX.sum(0).reshape(1, -1)
            np.divide(SXX, sum_pow, out=SXX)

            idx_sigma = np.logical_and(f >= 11, f <= 16)
            rel_power = SXX[idx_sigma].sum(0)

            text = 'spindles'
            spindles_counter_method1 = {}
            name = 0
            for item in rel_power:
                if item >= RELATIVE_SIGMA_POWER_THRESHOLD:
                    spindles_counter_method1['item' + str(name)] = [item]
                else:
                    name += 1
            if len(spindles_counter_method1) == 1:
                text = 'spindle'


            data_sigma = data_broad.copy()
            N = 20
            nyquist = sample_freq / 2
            Wn = 11 / nyquist
            sos = signal.iirfilter(N, Wn, btype='Highpass', output='sos')
            data_sigma = signal.sosfiltfilt(sos, data_sigma)

            Wn = 16 / nyquist
            sos = signal.iirfilter(N, Wn, btype='lowpass', output='sos')
            data_sigma = signal.sosfiltfilt(sos, data_sigma)

            dur = 0.3
            halfdur = dur / 2
            total_dur = len(data_sigma) / sample_freq
            last = len(data_sigma) - 1
            step = 0.1

            len_out = int(len(data_sigma) / (step * sample_freq))
            out = np.zeros(len_out)
            tt = np.zeros(len_out)

            for i, j in enumerate(np.arange(0, total_dur, step)):
                beg = int(max(0, ((j - halfdur) * sample_freq)))
                end = int(min(last, ((j + halfdur) * sample_freq)))
                tt[i] = (np.column_stack((beg, end)).mean(1) / sample_freq)
                win1 = data_sigma[beg:end]
                win2 = data_broad[beg:end]
                out[i] = stats.pearsonr(win1, win2)[0]

            moving_correlation = out
            interop = interp1d(tt, moving_correlation, kind='cubic', bounds_error=False,
                               fill_value=0, assume_sorted=True)

            tt = np.arange(data_sigma.size) / sample_freq

            moving_correlation = interop(tt)

            spindles_counter_method2 = {}
            name = 0
            for item in out:
                if item >= MOVING_CORRELATION_THRESHOLD:
                    spindles_counter_method2['item' + str(name)] = [item]
                else:
                    name += 1
            if len(spindles_counter_method2) == 1:
                text = 'spindle'

            tt2 = np.zeros(len_out)
            moving_rms_final = np.zeros(len_out)

            def _rms(x):
                n = x.size
                ms = 0
                for iii in range(n):
                    ms += x[iii] ** 2
                ms /= n
                return np.sqrt(ms)

            for i, j in enumerate(np.arange(0, total_dur, step)):
                beg = int(max(0, ((j - halfdur) * sample_freq)))
                end = int(min(last, ((j + halfdur) * sample_freq)))
                tt2[i] = (np.column_stack((beg, end)).mean(1) / sample_freq)
                win3 = data_sigma[beg:end]
                moving_rms_final[i] = _rms(win3)

            interop1 = interp1d(tt2, moving_rms_final, kind='cubic', bounds_error=False,
                                fill_value=0, assume_sorted=True)

            tt2 = np.arange(data_sigma.size) / sample_freq

            moving_rms_final = interop1(tt2)

            def trim_both_std(x, cut=0.10):
                x = np.asarray(x)
                n = x.shape[-1]
                lowercut = int(cut * n)
                uppercut = n - lowercut
                atmp = np.partition(x, (lowercut, uppercut - 1), axis=-1)
                sl = slice(lowercut, uppercut)
                return np.nanstd(atmp[..., sl], ddof=1, axis=-1)

            trimmed_std = trim_both_std(moving_rms_final, cut=0.025)
            thresh_rms = moving_rms_final.mean() + MOVING_RMS_THRESHOLD * trimmed_std

            spindles_counter_method3 = {}
            name = 0
            for item in moving_rms_final:
                if item >= thresh_rms:
                    spindles_counter_method3['item' + str(name)] = [item]
                else:
                    name += 1
            if len(spindles_counter_method3) == 1:
                text = 'spindle'

            idx_absolute = (abs_sig_pow >= ABSOLUTE_SIGMA_POWER_THRESHOLD).astype(int)
            idx_rel_pow = (rel_power >= 0.2).astype(int)
            idx_mcorr = (moving_correlation >= 0.69).astype(int)
            idx_mrms = (moving_rms_final >= thresh_rms).astype(int)

            idx_sum = (idx_absolute + idx_mrms + idx_mcorr + idx_rel_pow).astype(int)
            w = int(0.1 * sample_freq)
            idx_sum = np.convolve(idx_sum, np.ones(w) / w, mode='same')

            spindles_counter_method4 = {}
            name = 0
            for sssd in idx_sum:
                if sssd > THRESHOLD_FOR_GOOD_STUFF:
                    spindles_counter_method4['item' + str(name)] = [sssd]
                else:
                    name += 1
            if len(spindles_counter_method4) == 1:
                text = 'spindle'

            where_sp = np.where(idx_sum > THRESHOLD_FOR_GOOD_STUFF)[0]

            def merge(index, sf):
                min_distance = DISTANCE_BETWEEN_SPINDLES / 1000. * sf
                idx_diff = np.diff(index)
                condition = idx_diff > 1
                idx_distance = np.where(condition)[0]
                distance = idx_diff[condition]
                bad = idx_distance[np.where(distance < min_distance)[0]]

                if len(bad) > 0:
                    fill = np.hstack([np.arange(index[j] + 1, index[j + 1])
                                      for i, j in enumerate(bad)])
                    f_index = np.sort(np.append(index, fill))
                    return f_index
                else:
                    return index

            where_sp = merge(where_sp, sample_freq)
            sp = np.split(where_sp, np.where(np.diff(where_sp) != 1)[0] + 1)
            idx_start_end = np.array([[k[0], k[-1]] for k in sp]) / sample_freq
            sp_start, sp_end = idx_start_end.T
            sp_dur = sp_end - sp_start

            output = []
            for useless in range(len(sp_start)):
                if sp_dur[useless] > MAX_TIME_FOR_SPINDLES:
                    continue
                elif sp_dur[useless] < MIN_TIME_FOR_SPINDLES:
                    continue
                else:
                    output.append([sp_start[useless], sp_end[useless], sp_dur[useless]])
            arr_out = np.array(output)

            Marco_Ross_time = []
            for useless in range(len(sp_start)):
                if sp_dur[useless] > 1.5:
                    continue
                elif sp_dur[useless] < 0.5:
                    continue
                else:
                    Marco_Ross_time.append([sp_start[useless], sp_dur[useless]])
            spindles = np.array(Marco_Ross_time)

            mask = data_broad.copy()
            mask2 = data_broad.copy()
            SPINDLES_FINALE = pd.Series(mask, timesA)
            OVERSERIES2 = pd.Series(mask2, timesA)

            for fake in range(len(arr_out)):
                if fake == 0:
                    SPINDLES_FINALE[0:arr_out[fake][0]] = np.nan
                elif fake == (len(arr_out) - 1):
                    SPINDLES_FINALE[arr_out[fake - 1][1]:arr_out[fake][0]] = np.nan
                    SPINDLES_FINALE[arr_out[fake][1]:timesA.max()] = np.nan
                    break
                else:
                    SPINDLES_FINALE[arr_out[fake - 1][1]:arr_out[fake][0]] = np.nan

            for fake in range(len(sp_start)):
                if fake == 0:
                    OVERSERIES2[0:sp_start[0]] = np.nan
                elif fake == len(sp_start) - 1:
                    OVERSERIES2[sp_end[fake - 1]:sp_start[fake]] = np.nan
                    OVERSERIES2[sp_end[fake]:timesA.max()] = np.nan
                    break
                else:
                    OVERSERIES2[sp_end[fake - 1]:sp_start[fake]] = np.nan

            if len(arr_out) == 1:
                text = 'spindle'

            return OVERSERIES2.dropna()


    def start(self,data_name,sample_rate):
        data_path = (Path(__file__).absolute().parents[1] / 'input' / data_name).__str__()
        data = self.detect_spindles(data_path, sample_rate)
        index = data.index
        last = index[0]
        LastPoint = index[0]
        result = []
        area = {}
        for i in range(1,len(index)):
            if index[i] - last > 0.1:
                result.append([LastPoint,last])
                LastPoint = index[i]
            last = index[i]
        area['area'] = result
        file = open("newdata_tool.json","w")
        json.dump(area,file)
        file.close()
