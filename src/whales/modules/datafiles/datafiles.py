from whales.modules.formatters.formatters import Formatter
from whales.modules.module import Module


class Datafile(Module):
    def __init__(self, logger=None):
        super(Datafile, self).__init__(logger)
        self.metadata = {}
        self._data = None
        self.file_name = None
        self.formatter = None

    def load_data(self, file_name: str, formatter):
        """Do not actually load the data. Instead, save the access information."""
        self.file_name = file_name
        self.formatter = formatter
        self.metadata = formatter.read_metadata(file_name)
        return self

    def save_data(self, file_name: str, formatter):
        """Save the data in self.data into specified file_name with specified formatter and also write the metadata.
        If no data has changed, read data from the Datafile and write it in the file."""
        formatter.write(file_name, self.data)
        formatter.write_metadata(file_name, self.metadata)

    @property
    def data(self):
        return self._data or self.formatter.read(self.file_name)

    @data.setter
    def data(self, data):
        self._data = data
