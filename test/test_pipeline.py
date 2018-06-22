import sys, os
import pytest

from whales.modules.pipelines.parsers import NecessaryParameterAbsentError, UnexpectedParameterError, \
    UnexpectedTypeError

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

from glob import glob
from whales.utilities.testing import get_5_file_names, get_labeled
from whales.modules.pipelines.whale_detector import WhaleDetector


class TestWhalesDetectorPipeline:
    def test_missing_necessary_parameters(self):
        p = WhaleDetector()

        missing_parameters = """
    {
        "input_data": [
            {
                "file_name": "demo_data.h5",
                "data_file": "time_series",
                "formatter": "hdf5"
            }
        ],
        "output_directory": "./demo"
    }
        """

        with pytest.raises(NecessaryParameterAbsentError):
            p.load_parameters(missing_parameters)

    def test_extra_parameters(self):
        p = WhaleDetector()

        extra_parameters = """
    {
        "pipeline_type": "whale_detector",
        "input_data": [
            {
                "file_name": "demo_data.h5",
                "data_file": "time_series",
                "formatter": "hdf5"
            }
        ],
        "output_directory": "./demo",
        "temperature": 100.0
    }
        """

        with pytest.raises(UnexpectedParameterError):
            p.load_parameters(extra_parameters)

    def test_wrong_parameters(self):
        p = WhaleDetector()

        wrong_format_parameters = """
    {
        "pipeline_type": "whale_detector",
        "output_directory": "./demo",
        "input_data": [
            "demo_data.hdf5",
            "demo_data.aiff"
        ]
    }
        """

        with pytest.raises(UnexpectedTypeError):
            p.load_parameters(wrong_format_parameters)

    def test_whales_pipeline(self):
        _ = get_5_file_names()
        p = WhaleDetector()
        [os.remove(file) for file in glob("*.aif")]
        get_labeled()
        parameters = """{
            "pipeline_type": "whale_detector",
            "input_data": [
                {
                    "file_name": "*.aif",
                    "data_file": "audio",
                    "formatter": "aif"
                }
            ],
            "input_labels": [{
                "labels_file": "*.csv",
                "labels_formatter": "csv"
            }],
            "features_extractors": [
                {
                    "method": "skewness"
                }
            ],
            "output_directory": "./demo",
            "data_set_type": {
                "method": "files_fold"
            },
            "pre_processing": [
                {
                    "method": "scale"
                }
            ],
            "performance_indicators": [
                {
                    "method": "accuracy"
                }
            ],
            "machine_learning": {
                "method": "svm",
                "type": "supervised"
            }
        }"""
        p.load_parameters(parameters)
        p.initialize()
        p.start()
        assert "data_set" in p.results
        assert len(p.results["data_set"].datafiles) > 0
        assert "features_extractors" in p.results
        assert len(p.results["features_extractors"]) > 0
        assert "performance_indicators" in p.results
        assert len(p.results["performance_indicators"]) > 0
        assert False
