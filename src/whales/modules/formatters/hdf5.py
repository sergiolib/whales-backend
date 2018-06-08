from whales.modules.formatters.formatters import Formatter


class HDF5Formatter(Formatter):
    def read(self, filename):
        pass

    def write(self, filename):
        pass

    def read_metadata(self, metadata_filename):
        pass


PipelineFormatter = HDF5Formatter
