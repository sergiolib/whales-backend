import aifc
import numpy as np


def load_audio_test():
    """Return numpy array and sample rate of .aif audio file """
    fname = './data/ballena_bw_ruido_001_PU145_20120209_091500.aif'
    audio = aifc.open(fname, mode='r')
    nframes = audio.getnframes()
    rate = audio.getframerate()
    strData = audio.readframes(nframes)
    audio.close()
    ndarray = np.fromstring(strData, np.short).byteswap().astype(np.float64)
    return ndarray, rate