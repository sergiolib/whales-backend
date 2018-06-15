from whales.modules.formatters.formatters import Formatter
from whales.modules.module import Module


class Datafile(Module):
    def __init__(self, logger=None):
        super(Datafile, self).__init__(logger)
        self.data = None
        self.metadata = {}
        self.file_name = None
        self.formatter = None

    def load_data(self, file_name: str, formatter: Formatter):
        """Do not actually load the data. Instead, save the access information."""
        self.file_name = file_name
        self.formatter = formatter
        self.metadata = formatter.read_metadata(file_name)
        return self

    def save_data(self, file_name: str, formatter: Formatter):
        """Save the data in self.data into specified file_name with specified formatter and also write the metadata.
        If no data has been changed, read data from the Datafile and write it in the file."""
        self.data = self.data or self.formatter.read(self.file_name)
        formatter.write(file_name, self.data)
        formatter.write_metadata(file_name, self.data)  # TODO
