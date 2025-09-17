import numpy as np
import pandas as pd


def load_data(fn):
    '''
    fn을 읽어서 2차원 numpy 배열로 return

    return fft_abs_data (# of data, 320)
    '''
    ext = fn.split('.')[-1]
    if ext == 'csv':
        return load_csv(fn)

    elif ext == 'dat':
        return load_dat(fn)

    else:
        raise NotImplementedError()


def load_csv(fn):
    '''
    csv 포맷의 fn을 읽어서 2차원 numpy 배열로 return

    return fft_abs_data (# of data, 320)
    '''

    df = pd.read_csv(fn, encoding='cp949')
    sensor_data_df  = df['SENSOR_DATA'].str.split('|', expand=True)
    fft_abs_data = sensor_data_df.astype(np.float32).to_numpy()

    return fft_abs_data


def load_dat(fn):
    '''
    csv 포맷의 fn을 읽어서 2차원 numpy 배열로 return

    return fft_abs_data (# of data, 320)
    '''

    n_packet = 320
    fft_abs_data = np.memmap(fn, dtype=np.float32)
    fft_abs_data = np.reshape(fft_abs_data, [-1, n_packet])

    return np.array(fft_abs_data)