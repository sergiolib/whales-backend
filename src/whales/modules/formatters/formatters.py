class Formatter:
    def read(self, filename):
        raise NotImplementedError

    def write(self, filename):
        raise NotImplementedError

    def read_metadata(self, metadata_filename):
        raise NotImplementedError
