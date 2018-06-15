from urllib.request import urlretrieve

from whales.modules.datafiles.audio import AudioDatafile
from whales.modules.formatters.aif import AIFFormatter


class TestAudioDatafiles:
    @staticmethod
    def get_filename(desired_path=None):
        """Download and place a working AIFF sample file to perform the tests"""
        desired_path = desired_path or "ballena_bw_ruido_002_PU145_20120330_121500.aif"
        url = "https://www.cec.uchile.cl/~sliberman/ballena_bw_ruido_002_PU145_20120330_121500.aif"
        urlretrieve(url, desired_path)
        return desired_path

    def test_load(self):
        filename = self.get_filename()
        df = AudioDatafile()
        df.load_data(filename, AIFFormatter)
