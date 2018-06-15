import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

from src.whales.modules.preprocessing.scale import Scale
from utilities import load_audio_test
import numpy as np


def test_scale():
    data, rate = load_audio_test()
    p = Scale()
    result = p.transform(data=data)
    assert p.description != ""
    assert result.shape[0] == data.shape[0]
    assert result.dtype == np.float64

