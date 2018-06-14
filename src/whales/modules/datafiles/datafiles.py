from whales.modules.formatters.formatters import Formatter
from whales.modules.formatters.json import JSONFormatter
from whales.modules.module import Module


class Datafile(Module):
    def __init__(self, logger=None):
        super(Datafile, self).__init__(logger)
        self.data = None
        self.metadata = {}
        self.file_name = None
        self.formatter = None

    def load_data(self, file_name: str, formatter: Formatter, metadata_json_file=None):
        """Do not actually load the data. Instead, save the access information."""
        self.file_name = file_name
        self.formatter = formatter
        if metadata_json_file is None:
            self.metadata = formatter.read_metadata(file_name)
        else:
            self.metadata = JSONFormatter.read(metadata_json_file)

    def save_data(self, file_name: str, formatter: Formatter, metadata_json_file=None):
        """Save the data in self.data into specified file_name with specified formatter and also write the metadata.
        If no data has been changed, read data from the Datafile and write it in the file."""
        self.data = self.data or self.formatter.read(self.file_name)
        formatter.write(file_name, self.data)
        if metadata_json_file:
            JSONFormatter.write(metadata_json_file, self.metadata)

