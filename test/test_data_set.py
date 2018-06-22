from whales.utilities.testing import get_file_name, get_5_file_names
from whales.modules.data_files.audio import AudioDataFile
from whales.modules.data_sets.files_fold import FilesFoldDataSet
from whales.modules.formatters.aif import AIFFormatter


class TestFilesFoldDataSet:
    def test_add_remove_datafile(self):
        filename = get_file_name()
        df = AudioDataFile()
        df.load_data(filename,
                     formatter=AIFFormatter())
        ds = FilesFoldDataSet()
        ds.add_datafile(df)
        assert len(ds.datafiles) == 1
        ds.remove_datafile(df)
        assert len(ds.datafiles) == 0

    def test_get_training_testing_validation_set(self):
        ds = FilesFoldDataSet()
        file_names = get_5_file_names()
        for filename in file_names:
            ds.add_datafile(AudioDataFile().load_data(filename,
                                                      formatter=AIFFormatter()))
        assert len(ds.datafiles) == 5
        training = ds.get_training()
        testing = ds.get_testing()
        validation = ds.get_validation()
        expected_iterations = ds.iterations
        i = 0
        for tr, te, val in zip(training, testing, validation):
            i += 1
        assert expected_iterations == i

    def test_get_training_testing_set(self):
        ds = FilesFoldDataSet()
        ds.parameters["validation"] = False  # Disable validation set generation
        file_names = get_5_file_names()
        for filename in file_names:
            ds.add_datafile(AudioDataFile().load_data(filename,
                                                      formatter=AIFFormatter()))
        assert len(ds.datafiles) == 5
        training = ds.get_training()
        testing = ds.get_testing()
        expected_iterations = ds.iterations
        i = 0
        for tr, te in zip(training, testing):
            i += 1
        assert expected_iterations == i
