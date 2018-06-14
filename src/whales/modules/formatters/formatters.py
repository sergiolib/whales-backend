class Formatter:
    @staticmethod
    def read(filename, **kwargs):
        raise NotImplementedError

    @staticmethod
    def write(filename, data, **kwargs):
        raise NotImplementedError

    @staticmethod
    def read_metadata(metadata_filename, **kwargs):
        raise NotImplementedError

    @staticmethod
    def write_metadata(metadata_filename, metadata, **kwargs):
        raise NotImplementedError
