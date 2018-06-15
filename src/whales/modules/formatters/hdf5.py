from whales.modules.formatters.formatters import Formatter
from whales.modules.formatters.json import JSONMetadataMixin


class HDF5Formatter(JSONMetadataMixin,
                    Formatter):
    @staticmethod
    def read(filename):
        # TODO
        pass

    @staticmethod
    def write(filename, data):
        # TODO
        pass



PipelineFormatter = HDF5Metadata
