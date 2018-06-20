from whales.modules.module import Module


class Formatter(Module):
    def read(self, filename):
        raise NotImplementedError

    def write(self, filename, data):
        raise NotImplementedError

    def read_metadata(self, metadata_filename):
        raise NotImplementedError

    def write_metadata(self, metadata_filename, metadata):
        raise NotImplementedError
