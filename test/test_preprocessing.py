import sys, os

from whales.modules.data_files.audio import AudioDataFile
from whales.modules.formatters.aif import AIFFormatter

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

from src.whales.modules.pre_processing.scale import Scale
from utilities import get_file_name
import numpy as np


def test_scale():
    file_name = get_file_name()
    df = AudioDataFile().load_data(file_name,
                                   formatter=AIFFormatter())
    data = df.data.values.ravel()
    data = data / abs(data).max()
    p = Scale()
    result = p.transform(data=data)
    assert p.description != ""
    assert result.shape[0] == data.shape[0]
    assert result.dtype == np.float64

