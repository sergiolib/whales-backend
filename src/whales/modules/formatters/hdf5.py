import pandas as pd

from whales.modules.formatters.formatters import Formatter
from whales.modules.formatters.json import JSONMetadataMixin


class HDF5Formatter(JSONMetadataMixin,
                    Formatter):
    def read(self, filename, key="whales_data"):
        return pd.read_hdf(path_or_buf=filename, key=key, mode='r')

    def write(self, filename, data: pd.DataFrame, key="whales_data"):
        data.to_hdf(path_or_buf=filename, key=key, mode="w")


PipelineFormatter = HDF5Formatter
