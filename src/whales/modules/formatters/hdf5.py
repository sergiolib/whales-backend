from whales.modules.formatters.formatters import Formatter
from whales.modules.formatters.json import JSONFormatterMixin


class HDF5Formatter(JSONFormatterMixin,
                    Formatter):
    @staticmethod
    def read(filename):
        # TODO
        pass

    @staticmethod
    def write(filename, data):
        # TODO
        pass



PipelineFormatter = HDF5Formatter
