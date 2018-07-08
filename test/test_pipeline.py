import sys, os
import pytest
from pandas._libs import json

from whales.modules.pipelines.getters import get_available_pipeline_types
from whales.modules.pipelines.parsers import NecessaryParameterAbsentError, UnexpectedParameterError, \
    UnexpectedTypeError
from whales.modules.supervised.svm import SVM

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

from glob import glob
from whales.utilities.testing import get_5_file_names, get_labeled
from whales.modules.pipelines.whale_detector import WhaleDetectorForTests


class TestWhalesDetectorPipeline:
    def test_missing_necessary_parameters(self):
        p = WhaleDetectorForTests()

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
        p = WhaleDetectorForTests()

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
        p = WhaleDetectorForTests()

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
        p = WhaleDetectorForTests()
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
                },
                {
                    "method": "range"
                },
                {
                    "method": "mfcc",
                    "parameters": {
                        "n_components": 25
                    }
                }
            ],
            "output_directory": "./demo",
            "data_set_type": {
                "method": "windows_fold"
            },
            "pre_processing": [
                {
                    "method": "scale"
                },
                {
                    "method": "sliding_windows",
                    "parameters": {
                        "window_width": "60s",
                        "overlap": 0.3,
                        "labels_treatment": "mode"
                    }
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
        assert "features_extractors" in p.results
        assert len(p.results["features_extractors"]) > 0
        assert "performance_indicators" in p.results
        assert len(p.results["performance_indicators"]) > 0
        assert type(p.results["ml_method"]) is SVM

    def test_train_whales_pipeline(self):
        parameters = """{
            "pipeline_type": "train_whale_detector",
            "input_data": [
                {
                    "file_name": "/Volumes/HDD/Dropbox/Detector ballena azul/supervised_version/database/etiquetas/audios/ballena_bw_ruido_001_PU145_20120209_091500.aif",
                    "data_file": "audio",
                    "formatter": "aif"
                },
                {
                    "file_name": "/Volumes/HDD/Dropbox/Detector ballena azul/supervised_version/database/etiquetas/audios/ballena_bw_ruido_002_PU145_20120330_121500.aif",
                    "data_file": "audio",
                    "formatter": "aif"
                }
            ],
            "input_labels": [{
                "labels_file": "/Volumes/HDD/Dropbox/Detector ballena azul/supervised_version/database/etiquetas/csv/ballena_bw_ruido_00*.csv",
                "labels_formatter": "csv"
            }],
            "features_extractors": [
                {
                    "method": "skewness"
                },
                {
                    "method": "range"
                },
                {
                    "method": "mfcc",
                    "parameters": {
                        "n_components": 25
                    }
                }
            ],
            "output_directory": "./demo",
            "pre_processing": [
                {
                    "method": "scale"
                },
                {
                    "method": "sliding_windows",
                    "parameters": {
                        "window_width": "60s",
                        "overlap": 0.3,
                        "labels_treatment": "mode"
                    }
                }
            ],
            "machine_learning": {
                "method": "svm",
                "type": "supervised"
            }
        }"""
        js_parameters = json.loads(parameters)
        available_pipelines = get_available_pipeline_types()
        assert "pipeline_type" in js_parameters
        p = available_pipelines[js_parameters["pipeline_type"]]()
        p.load_parameters(parameters)
        p.initialize()
        p.start()
        assert "features_extractors" in p.results
        assert len(p.results["features_extractors"]) > 0
        assert type(p.results["ml_method"]) is SVM

    def test_predict_whales_pipeline(self):
        # self.test_train_whales_pipeline()
        parameters = """{
            "pipeline_type": "predict_whale_detector",
            "input_data": [
                {
                    "file_name": "/Volumes/HDD/Dropbox/Detector ballena azul/supervised_version/database/etiquetas/audios/ballena_bw_ruido_003_PU145_20120502_213000.aif",
                    "data_file": "audio",
                    "formatter": "aif"
                },
                {
                    "file_name": "/Volumes/HDD/Dropbox/Detector ballena azul/supervised_version/database/etiquetas/audios/ballena_bw_ruido_004_PU145_20120502_221500.aif",
                    "data_file": "audio",
                    "formatter": "aif"
                }
            ],
            "input_labels": [{
                "labels_file": "/Volumes/HDD/Dropbox/Detector ballena azul/supervised_version/database/etiquetas/csv/ballena_bw_ruido_00*.csv",
                "labels_formatter": "csv"
            }],
            "features_extractors": [
                {
                    "method": "skewness"
                },
                {
                    "method": "range"
                },
                {
                    "method": "mfcc",
                    "parameters": {
                        "n_components": 25
                    }
                }
            ],
            "output_directory": "./demo",
            "pre_processing": [
                {
                    "method": "scale"
                },
                {
                    "method": "sliding_windows",
                    "parameters": {
                        "window_width": "60s",
                        "overlap": 0.3,
                        "labels_treatment": "mode"
                    }
                }
            ],
            "performance_indicators": [
                {
                    "method": "accuracy"
                },
                {
                    "method": "confusion_matrix",
                    "parameters": {
                        "plot": true
                    }
                }
            ],
            "machine_learning": {
                "method": "svm",
                "type": "supervised"
            }
        }"""
        js_parameters = json.loads(parameters)
        available_pipelines = get_available_pipeline_types()
        assert "pipeline_type" in js_parameters
        p = available_pipelines[js_parameters["pipeline_type"]]()
        p.load_parameters(parameters)
        p.initialize()
        p.start()
        assert "features_extractors" in p.results
        assert len(p.results["features_extractors"]) > 0
        assert "performance_indicators" in p.results
        assert len(p.results["performance_indicators"]) > 0
        assert type(p.results["ml_method"]) is SVM
