import sys, os
import numpy as np
import pytest

from whales.modules.data_files.feature import FeatureDataFile, AudioSegments
from whales.modules.features_extractors.energy import Energy
from whales.modules.features_extractors.kurtosis import Kurtosis

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

from whales.utilities.testing import get_file_name, get_labeled, get_labeled_txt, get_5_file_names
from whales.modules.data_files.audio import AudioDataFile
from whales.modules.formatters.aif import AIFFormatter
from whales.modules.formatters.hdf5 import HDF5Formatter
from whales.modules.labels_formatters.csv import CSVLabelsFormatter
from whales.modules.labels_formatters.txt import TXTLabelsFormatter


class TestAudioDataFiles:
    def test_load(self):
        filename = get_file_name()
        df = AudioDataFile()
        # No attribute data
        assert str(df) == "AudioDataFile"
        with pytest.raises(RuntimeError):
            df.duration
        with pytest.raises(RuntimeError):
            df.start_time
        with pytest.raises(RuntimeError):
            df.end_time
        df.load(filename,
                formatter=AIFFormatter())
        assert df.duration.seconds > 0
        assert str(df) == "AudioDataFile (0 days 00:14:59.999500)"

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

    def test_add_windows(self):
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

        # No windows added yet
        assert df.parameters["number_of_windows"] == 0
        d = df.get_window(0)
        assert all(d == df.data)

        # Add windows
        # First an incorrect one
        with pytest.raises(AttributeError):
            df.add_window(st1 + 100, en1 - 1)

        df.add_window(st1, en1)  # Length 6
        df.add_window(st2, en2)  # Length 4
        assert df.parameters["number_of_windows"] == 2
        assert str(df) == "AudioDataFile (2 windows)"

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
        wdf = df.get_windows_data_frame()
        assert len(wdf) == 2
        w1, l1 = df.get_window(0)
        w2, l2 = df.get_window(1)
        assert len(w1) == 6
        assert len(w2) == 4

        # Out of range index
        assert df.get_window(100) is None

        # Other label treatments
        df.parameters["labels_treatment"] = "mode"
        df.get_window(0)
        df.parameters["labels_treatment"] = "mean"
        _, l1mn = df.get_window(0)
        assert type(l1mn) is float
        with pytest.raises(ValueError):
            df.parameters["labels_treatment"] = "fourier"
            # There is actually no labels treatment called fourier
            df.get_window(0)  # Raises value error

    def test_copy_dataframe(self):
        filename_data, filename_labels = get_labeled_txt()
        df = AudioDataFile().load(filename_data,
                                  formatter=AIFFormatter())
        df.load_labels(filename_labels,
                       labels_formatter=TXTLabelsFormatter(),
                       label="whale")
        new_df = AudioDataFile(df)
        np.testing.assert_equal(new_df.metadata["labels"], df.metadata["labels"])

    def test_concatenate_data_files(self):
        fns = get_5_file_names()
        dfs = []
        for d in fns:
            dfs.append(AudioDataFile().load(d, formatter=AIFFormatter()))
        big_df = AudioDataFile().concatenate(dfs)
        assert big_df.duration.seconds >= sum([d.duration.seconds for d in dfs])

    def test_sampling_rate(self):
        filename = get_file_name()
        df = AudioDataFile()
        assert df.sampling_rate is None
        df.load(filename,
                formatter=AIFFormatter())
        assert df.sampling_rate > 0


class TestFeatureDataFiles:
    def load_feature_datafile(self):
        filename_data, filename_labels = get_labeled_txt()
        df = AudioDataFile().load(filename_data,
                                  formatter=AIFFormatter())
        df.load_labels(filename_labels,
                       labels_formatter=TXTLabelsFormatter(),
                       label="whale")
        # Make 5 windows
        st = df.start_time
        step = df.duration / 5
        out = AudioSegments()
        for i in range(5):
            df.add_window(st + i * step, st + (i + 1) * step)
            out.add_segment(*df.get_window(i))
        df = out

        # Get features
        f1 = Energy()
        f1.parameters["data"] = df
        f2 = Kurtosis()
        f2.parameters["data"] = df
        f1 = f1.transform()
        f2 = f2.transform()

        return FeatureDataFile().concatenate([f1, f2])

    def test_save_load(self):
        df = self.load_feature_datafile()

        # Test repr
        assert str(df) == "FeatureDataFile (5 samples x 2 columns)"
        new_df = FeatureDataFile()
        assert str(new_df) == "FeatureDataFile"

        # Test copy
        new_df = FeatureDataFile(df)
        assert df.data.equals(new_df.data)

        # Test save
        df.save("tmp.h5", formatter=HDF5Formatter())

        new_df = FeatureDataFile().load("tmp.h5", formatter=HDF5Formatter())
        assert df.data.equals(new_df.data)

    def test_segments_repr(self):
        filename_data, filename_labels = get_labeled_txt()
        df = AudioDataFile().load(filename_data,
                                  formatter=AIFFormatter())
        df.load_labels(filename_labels,
                       labels_formatter=TXTLabelsFormatter(),
                       label="whale")
        # Make 5 windows
        st = df.start_time
        step = df.duration / 5
        out = AudioSegments()
        assert str(out) == "AudioSegments"

        for i in range(5):
            df.add_window(st + i * step, st + (i + 1) * step)
            out.add_segment(*df.get_window(i))

        assert str(out) == "AudioSegments (5 segments)"
