from whales.modules.data_files.data_files import DataFile


class TimeSeriesDataFile(DataFile):
    @property
    def sampling_rate(self):
        if hasattr(self, "data") and self.data is not None:
            ind = self.data.index
            return 1e6 / (ind[1].microsecond - ind[0].microsecond)
        else:
            return None
