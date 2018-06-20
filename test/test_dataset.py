from test_datafiles import TestAudioDatafiles
from utilities import get_5_filenames
from whales.modules.datafiles.audio import AudioDatafile
from whales.modules.datasets.files_fold import FilesFoldDataSet
from whales.modules.formatters.aif import AIFFormatter


class TestFilesFoldDataSet:
    def test_add_remove_datafile(self):
        df = TestAudioDatafiles().test_load()
        ds = FilesFoldDataSet()
        ds.add_datafile(df)
        assert len(ds.datafiles) == 1
        ds.remove_datafile(df)
        assert len(ds.datafiles) == 0

    def test_get_training_testing_validation_set(self):
        ds = FilesFoldDataSet()
        file_names = get_5_filenames()
        for filename in file_names:
            ds.add_datafile(AudioDatafile().load_data(filename,
                                                      formatter=AIFFormatter()))
        assert len(ds.datafiles) == 5
        training = ds.get_training()
        testing = ds.get_testing()
        validation = ds.get_validation()
        used_te = []
        used_val = []
        for tr, te, val in zip(training, testing, validation):
            # This iteration's test data should not have been used before
            assert te.data.columns[0] not in used_te
            # This iteration's validation data should not have been used before
            assert val.data.columns[0] not in used_val
            # This iteration's test data should not be present in training set
            assert te.data.columns[0] not in tr.data.columns
            # This iteration's test data should not be present in validation set
            assert te.data.columns[0] not in val.data.columns
            # This iteration's validation data should not be present in training set
            assert val.data.columns[0] not in tr.data.columns
            # This iteration's validation data should not be present in testing set
            assert val.data.columns[0] not in te.data.columns

            used_te.append(te.data.columns[0])  # Append this test data into previously used test data
            used_val.append(val.data.columns[0])  # Append this validation data into previously used validation data

    def test_get_training_testing_set(self):
        ds = FilesFoldDataSet()
        ds.parameters["validation"] = False  # Disable validation set generation
        file_names = get_5_filenames()
        for filename in file_names:
            ds.add_datafile(AudioDatafile().load_data(filename,
                                                      formatter=AIFFormatter()))
        assert len(ds.datafiles) == 5
        training = ds.get_training()
        testing = ds.get_testing()
        used_te = []
        for tr, te in zip(training, testing):
            # This iteration's test data should not have been used before
            assert te.data.columns[0] not in used_te
            # This iteration's test data should not be present in training set
            assert te.data.columns[0] not in tr.data.columns

            used_te.append(te.data.columns[0])  # Append this test data into previously used test data
