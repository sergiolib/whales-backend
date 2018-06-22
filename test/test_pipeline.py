import sys, os
import pytest

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

from glob import glob
from whales.utilities.testing import get_5_file_names, get_labeled
from whales.modules.pipelines.whale_detector import WhaleDetector


class TestWhalesDetectorPipeline:
    def test_load_parameters(self):
        """Test that loading set of parameters works the way it is expected"""
        p = WhaleDetector()

        demo_parameters = """
    {
        "pipeline_type": "whale_detector",
        "input_data": [
            {
                "file_name": "demo_data.h5",
                "data_file": "time_series",
                "formatter": "hdf5"
            }
        ],
        "input_labels": [{
            "labels_file": "demo_labels.txt",
            "labels_formatter": "txt"
        }],
        "pre_processing": [
            {
                "method": "scale",
                "parameters": {}
            }
        ],
        "features_extractors": [
            {
                "method": "identity",
                "parameters": {}
            }
        ],
        "machine_learning": {
            "type": "unsupervised",
            "method": "kmeans",
            "parameters": {}
        },
        "performance_indicators": [
            {
                "method": "accuracy",
                "parameters": {}
            }
        ],
        "output_directory": "./demo",
        "data_set_type": {
            "method": "files_fold"
        }
    }
        """

        p.load_parameters(demo_parameters)
        assert type(p.parameters["machine_learning"]) is dict
        assert p.parameters["performance_indicators"][0]["method"] == "accuracy"

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

        with pytest.raises(ValueError):
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

        with pytest.raises(ValueError):
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
        ],
    }
        """

        with pytest.raises(ValueError):
            p.load_parameters(wrong_format_parameters)

    def test_whales_pipeline(self):
        _ = get_5_file_names()
        p = WhaleDetector()
        [os.remove(file) for file in glob("*.aif")]
        labeled_fns = get_labeled()
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
            "output_directory": "./demo"
        }"""
        p.load_parameters(parameters)
        p.initialize()
        p.start()
        ds = p.results["data_set"]
        tr = ds.get_training()
        for t in tr:
            return
