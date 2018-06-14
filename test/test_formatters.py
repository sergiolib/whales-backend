import pandas as pd

from os import rename
from os.path import isfile
from urllib.request import urlretrieve
from whales.modules.formatters.aif import AIFFormatter
from whales.modules.formatters.json import JSONFormatter


class TestAIFFormatter:
    @staticmethod
    def get_filename(desired_path=None):
        """Download and place a working AIFF sample file to perform the tests"""
        desired_path = desired_path or "ballena_bw_ruido_002_PU145_20120330_121500.aif"
        url = "https://uc05bb4af8cba350208717dbc3c8.dl.dropboxusercontent.com/cd/0/get/AI3QzjmFBZeKNLSJKze7H6PoZQEC3LNSGYTJq1Fp6eePZCIqPGsJBBXT6krrSeP5wOjKnkHKT8gCIKpmtmADg2cJt1Ewaw4ORgI1FmftHO4YODLTIYctu8oaqD4wmoQ2W_W-UQWyU543tgVxTWjhy5TIUVSPPsxMk54M2PfAis0D0tGSmMyfpmRoEXzEa4__0AA/file?dl=1"
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


class TestJSONFormatter:
    def test_read(self):
        with open("sample_json.json", "w") as f:
            f.write('{"kind_of_file": "json_file", "list": ["Yes, this is a list", "of many elements"]}')
        filename = "sample_json.json"
        formatter = JSONFormatter()
        metadata = formatter.read(filename)
        assert len(metadata) == 2
        assert type(metadata["list"]) is list

    def test_write(self):
        filename = "sample_json.json"
        formatter = JSONFormatter()
        formatter.write(filename, {"Parameters": [1, 2, 3]})
        assert isfile(filename)
        with open(filename, "r") as f:
            content = f.read()
        content = content.replace("\n", "").replace(" ", "")
        assert content == '{"Parameters":[1,2,3]}'
