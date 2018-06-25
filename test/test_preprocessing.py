import sys, os

from whales.modules.pre_processing.sliding_windows import SlidingWindows

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import numpy as np
from whales.modules.data_files.audio import AudioDataFile
from whales.modules.formatters.aif import AIFFormatter
from whales.modules.pre_processing.scale import Scale
from whales.utilities.testing import get_file_name


def test_scale():
    file_name = get_file_name()
    df = AudioDataFile().load_data(file_name,
                                   formatter=AIFFormatter())
    data = df.data.values.ravel()
    data = data / abs(data).max()
    p = Scale()
    result = p.transform(data_file=data)
    assert p.description != ""
    assert result.shape[0] == data.shape[0]
    assert result.dtype == np.float64


def test_sliding_windows():
    filename = get_file_name()
    p = SlidingWindows()
    p.parameters["sliding_window_width"] = "13s"
    p.parameters["overlap"] = 0.12
    df = AudioDataFile()
    df.load_data(filename,
                 formatter=AIFFormatter())
    new_df = p.transform(df)
    new_df.get_sliding_windows_data_frame()
