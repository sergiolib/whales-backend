import sys, os

from whales.modules.pre_processing.sliding_windows import SlidingWindows

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

from whales.modules.data_files.audio import AudioDataFile
from whales.modules.formatters.aif import AIFFormatter
from whales.modules.pre_processing.scale import Scale
from whales.utilities.testing import get_file_name


def test_scale():
    file_name = get_file_name()
    df = AudioDataFile().load(file_name,
                                   formatter=AIFFormatter())
    p = Scale()
    p.private_parameters["data"] = df
    result = p.transform()
    assert p.description != ""
    assert type(result) is df.__class__


def test_sliding_windows():
    filename = get_file_name()
    p = SlidingWindows()
    p.parameters["sliding_window_width"] = "13s"
    p.parameters["overlap"] = 0.12
    df = AudioDataFile()
    df.load(filename,
                 formatter=AIFFormatter())
    p.private_parameters["data"] = df
    new_df = p.transform()
    new_df.get_windows_data_frame()
    assert p.description != ""
