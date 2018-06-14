import json


class JSONFormatterMixin():
    """Mixin for using where format doesn't have built in metadata storage"""
    @staticmethod
    def read_metadata(metadata_filename):
        return json.load(open(metadata_filename, mode="r"))

    @staticmethod
    def write_metadata(metadata_filename, metadata):
        json.dump(metadata, open(metadata_filename, mode="w"))
