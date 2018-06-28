import sys, os


myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

from whales.utilities.testing import get_file_name, get_labeled, get_labeled_txt
from whales.modules.data_files.audio import AudioDataFile
from whales.modules.formatters.aif import AIFFormatter
from whales.modules.formatters.hdf5 import HDF5Formatter
from whales.modules.labels_formatters.csv import CSVLabelsFormatter
from whales.modules.labels_formatters.txt import TXTLabelsFormatter


class TestAudioDataFiles:
    def test_load(self):
        filename = get_file_name()
        df = AudioDataFile()
        df.load_data(filename,
                     formatter=AIFFormatter())
        assert df.duration.seconds > 0

    def test_save(self):
        filename = get_file_name()
        df = AudioDataFile()
        df.load_data(filename,
                     formatter=AIFFormatter())
        df.save_data("tmp.h5", formatter=HDF5Formatter())
        df2 = AudioDataFile().load_data("tmp.h5", formatter=HDF5Formatter())
        assert df2.duration.seconds > 0

    def test_load_labels(self):
        filename_data, filename_labels = get_labeled()
        df = AudioDataFile().load_data(filename_data,
                                       formatter=AIFFormatter())
        df.load_labels(filename_labels,
                       labels_formatter=CSVLabelsFormatter(),
                       label="whale")
        assert [0, 1] in df.get_labeled_data().labels.unique()
        assert "whale" in df.name_label

    def test_load_labels_from_txt(self):
        filename_data, filename_labels = get_labeled_txt()
        df = AudioDataFile().load_data(filename_data,
                                       formatter=AIFFormatter())
        df.load_labels(filename_labels,
                       labels_formatter=TXTLabelsFormatter(),
                       label="whale")
        assert [0, 1] in df.get_labeled_data().labels.unique()
        assert "whale" in df.name_label
