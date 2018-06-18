import sys, os

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

from utilities import get_filename
from whales.modules.datafiles.audio import AudioDatafile
from whales.modules.formatters.aif import AIFFormatter
from whales.modules.formatters.hdf5 import HDF5Formatter


class TestAudioDatafiles:
    def test_load(self):
        filename = get_filename()
        df = AudioDatafile()
        df.load_data(filename, AIFFormatter)
        assert df.duration > 0
        return df

    def test_save(self):
        filename = get_filename()
        df = AudioDatafile()
        df.load_data(filename, AIFFormatter)
        df.save_data("tmp.h5", HDF5Formatter)
        df2 = AudioDatafile().load_data("tmp.h5", HDF5Formatter)
