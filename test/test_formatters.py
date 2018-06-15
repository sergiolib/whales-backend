import pandas as pd

from os import rename
from urllib.request import urlretrieve
from whales.modules.formatters.aif import AIFFormatter


class TestAIFFormatter:
    @staticmethod
    def get_filename(desired_path=None):
        """Download and place a working AIFF sample file to perform the tests"""
        desired_path = desired_path or "ballena_bw_ruido_002_PU145_20120330_121500.aif"
        url = "https://www.cec.uchile.cl/~sliberman/ballena_bw_ruido_002_PU145_20120330_121500.aif"
        urlretrieve(url, desired_path)
        return desired_path

    def test_read(self):
        filename = TestAIFFormatter.get_filename()
        formatter = AIFFormatter()
        series = formatter.read(filename)
        assert type(series) is pd.Series
        assert len(series) > 0
        assert type(series.index) is pd.DatetimeIndex

        new_filename = "same_file_new_name.aif"
        rename(filename, "same_file_new_name.aif")
        series = formatter.read(new_filename)
        assert type(series) is pd.Series
        assert len(series) > 0
        assert type(series.index) is pd.RangeIndex

    def test_read_metadata(self):
        filename = TestAIFFormatter.get_filename()
        formatter = AIFFormatter()
        metadata = formatter.read_metadata(filename)
        assert type(metadata) is dict
