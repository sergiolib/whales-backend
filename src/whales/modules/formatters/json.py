import json
from os.path import splitext

from whales.utilities.json import WhalesEncoder, WhalesDecoder


class JSONMetadataMixin:
    """Mixin for using where format doesn't have built in metadata storage"""
    def read_metadata(self, metadata_filename):
        metadata_filename = correct_metadata_filename(metadata_filename)
        return json.load(open(metadata_filename, mode="r"), cls=WhalesDecoder)

    def write_metadata(self, metadata_filename, metadata):
        metadata_filename = correct_metadata_filename(metadata_filename)
        json.dump(metadata, open(metadata_filename, mode="w"), cls=WhalesEncoder)


def correct_metadata_filename(metadata_filename):
    name, ext = splitext(metadata_filename)
    if ext != ".json":
        metadata_filename = name + ".json"
    return metadata_filename
