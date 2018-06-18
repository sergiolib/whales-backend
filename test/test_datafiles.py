import sys, os

from whales.modules.datafiles.datafiles import Datafile
from whales.modules.labels_formatters.csv import CSVLabelsFormatter

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

from utilities import get_filename, get_labeled
from whales.modules.datafiles.audio import AudioDatafile
from whales.modules.formatters.aif import AIFFormatter
from whales.modules.formatters.hdf5 import HDF5Formatter


class TestAudioDatafiles:
    def test_load(self):
        filename = get_filename()
        df = AudioDatafile()
        df.load_data(filename,
                     formatter=AIFFormatter)
        assert df.duration.seconds > 0
        return df

    def test_save(self):
        filename = get_filename()
        df = AudioDatafile()
        df.load_data(filename,
                     formatter=AIFFormatter)
        df.save_data("tmp.h5", HDF5Formatter)
        df2 = AudioDatafile().load_data("tmp.h5", HDF5Formatter)
        assert df2.duration.seconds > 0

    def test_mutate(self):
        filename = get_filename()
        df = Datafile()
        df.load_data(filename,
                     formatter=AIFFormatter)
        assert not hasattr(df, "duration")  # Duration exists only in audio datafile
        audio_df = AudioDatafile(df)
        assert audio_df.duration.seconds > 0

    def test_load_labels(self):
        filename_data, filename_labels = get_labeled()
        df = AudioDatafile().load_data(filename_data,
                                       formatter=AIFFormatter)
        df.load_labels(filename_labels, CSVLabelsFormatter, label="whale")
        assert [0, 1] in df.data.labels.unique()
        assert "whale" in df.name_label
