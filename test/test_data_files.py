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
        df.load(filename,
                     formatter=AIFFormatter())
        assert df.duration.seconds > 0

    def test_save(self):
        filename = get_file_name()
        df = AudioDataFile()
        df.load(filename,
                     formatter=AIFFormatter())
        df.save("tmp.h5", formatter=HDF5Formatter())
        df2 = AudioDataFile().load("tmp.h5", formatter=HDF5Formatter())
        assert df2.duration.seconds > 0

    def test_load_labels(self):
        filename_data, filename_labels = get_labeled()
        df = AudioDataFile().load(filename_data,
                                       formatter=AIFFormatter())
        df.load_labels(filename_labels,
                       labels_formatter=CSVLabelsFormatter(),
                       label="whale")
        assert [0, 1] in df.get_labeled_data().labels.unique()
        assert "whale" in df.name_label

    def test_load_labels_from_txt(self):
        filename_data, filename_labels = get_labeled_txt()
        df = AudioDataFile().load(filename_data,
                                       formatter=AIFFormatter())
        df.load_labels(filename_labels,
                       labels_formatter=TXTLabelsFormatter(),
                       label="whale")
        assert [0, 1] in df.get_labeled_data().labels.unique()
        assert "whale" in df.name_label

    def test_get_window(self):
        filename_data, filename_labels = get_labeled_txt()
        df = AudioDataFile().load(filename_data,
                                       formatter=AIFFormatter())
        df.load_labels(filename_labels,
                       labels_formatter=TXTLabelsFormatter(),
                       label="whale")
        st1 = df.data.index[0]
        en1 = df.data.index[5]
        st2 = df.metadata["labels"][0][0]
        en2 = st2 + 4
        df.add_window(st1, en1)  # Length 6
        df.add_window(st2, en2)  # Length 4
        assert df.parameters["number_of_windows"] == 2
        l = df.get_windows_data_frame()
        assert len(l) == 2
        w1, l1 = df.get_window(0)
        w2, l2 = df.get_window(1)
        assert len(w1) == 6
        assert len(w2) == 4
