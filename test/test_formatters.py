import sys, os

from utilities import get_filename

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import pandas as pd

from os import rename
from whales.modules.formatters.aif import AIFFormatter


class TestAIFFormatter:
    def test_read(self):
        filename = get_filename()
        formatter = AIFFormatter()
        df = formatter.read(filename)
        assert type(df) is pd.DataFrame
        assert len(df) > 0
        assert type(df.index) is pd.DatetimeIndex

        new_filename = "same_file_new_name.aif"
        rename(filename, "same_file_new_name.aif")
        df = formatter.read(new_filename)
        assert type(df) is pd.DataFrame
        assert len(df) > 0
        assert type(df.index) is pd.RangeIndex

    def test_read_metadata(self):
        filename = get_filename()
        formatter = AIFFormatter()
        metadata = formatter.read_metadata(filename)
        assert type(metadata) is dict
