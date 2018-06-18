from test_datafiles import TestAudioDatafiles
from whales.modules.datasets.kfold import KFoldDataSet


class TestKFoldDataset:
    def test_add_remove_datafile(self):
        df = TestAudioDatafiles().test_load()
        ds = KFoldDataSet()
        ds.add_datafile(df)
        assert len(ds.datafiles) == 1
        ds.remove_datafile(df)
        assert len(ds.datafiles) == 0
