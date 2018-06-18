import numpy as np
import pandas as pd
import aifc

from os.path import splitext, basename
from .formatters import Formatter


class AIFFormatter(Formatter):
    @staticmethod
    def read(filename: str, dtype=np.short, byteswapped=True):
        """Read the AIF file for the data and return a pandas Series"""
        # Open the file
        with aifc.open(filename, mode="r") as aiff:
            n_frames = aiff.getnframes()  # Number of samples in total
            rate = aiff.getframerate()  # Get the frame rate to label correctly the samples
            str_data = aiff.readframes(n_frames)  # Read the whole file

        # Transform from bytes to array
        ndarray = np.fromstring(str_data, dtype).byteswap()

        # Swap bytes (little endian -> bin endian conversion)
        if byteswapped:
            ndarray = ndarray.byteswap()

        # Get the index if possible
        try:
            no_ext = splitext(basename(filename))[0]
            date = no_ext.split("_")[-2]
            time = no_ext.split("_")[-1]
            year = date[:4]
            month = date[4:6]
            day = date[6:]
            hours = time[:2]
            minutes = time[2:4]
            seconds = time[4:]
            rng = pd.date_range(start=f'{year}/{month}/{day} {hours}:{minutes}:{seconds}',
                                periods=n_frames,
                                freq=f'{1e9//rate}ns')
        except Exception:
            rng = None
        return pd.Series(ndarray, index=rng, name=basename(filename)).to_frame()  # Return a DataFrame object

    @staticmethod
    def write(filename, data):
        raise NotImplementedError

    @staticmethod
    def read_metadata(metadata_filename):
        """Read the metadata filename (might be the same data file) and return a dictionary of metadata."""
        # Open the file
        metadata = {}
        with aifc.open(metadata_filename, mode="r") as aiff:
            metadata["num_frames"] = aiff.getnframes()
            metadata["frame_rate"] = aiff.getframerate()
            metadata["num_channels"] = aiff.getnchannels()
            metadata["compression_name"] = aiff.getcompname().decode("utf-8")
            metadata["compression_type"] = aiff.getcomptype().decode("utf-8")
            metadata["markers"] = aiff.getmarkers()
            metadata["sample_width"] = aiff.getsampwidth()
        return metadata

    @staticmethod
    def write_metadata(metadata_filename, metadata):
        raise NotImplementedError


PipelineFormatter = AIFFormatter
