import json
from os.path import splitext


class JSONMetadataMixin:
    """Mixin for using where format doesn't have built in metadata storage"""
    @staticmethod
    def read_metadata(metadata_filename):
        metadata_filename = correct_metadata_filename(metadata_filename)
        return json.load(open(metadata_filename, mode="r"))

    @staticmethod
    def write_metadata(metadata_filename, metadata):
        metadata_filename = correct_metadata_filename(metadata_filename)
        json.dump(metadata, open(metadata_filename, mode="w"))


def correct_metadata_filename(metadata_filename):
    name, ext = splitext(metadata_filename)
    if ext != ".json":
        metadata_filename = name + ".json"
    return metadata_filename
