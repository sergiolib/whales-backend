import sys, os

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

from whales.modules.pipelines.pipeline import get_available_features_extractors
from whales.modules.pipelines.pipeline import get_available_pre_processing
from whales.modules.pipelines.pipeline import get_available_pipeline_types
from whales.modules.pipelines.pipeline import Pipeline
from whales.modules.pipelines.pipeline import get_available_formatters
from whales.modules.pipelines.pipeline import get_available_datafiles
from whales.modules.pipelines.pipeline import get_available_performance_indicators
from whales.modules.pipelines.pipeline import get_available_unsupervised_methods
from whales.modules.pipelines.pipeline import get_available_supervised_methods
from whales.modules.pipelines.pipeline import get_available_semi_supervised_methods
from whales.modules.pipelines.pipeline import get_available_datasets


def test_get_available_feature_extractors():
    """Test that get_available_feature_extractors function works correctly"""
    fe = get_available_features_extractors()
    assert "identity" in fe


def test_get_available_pre_processing():
    """Test that get_available_feature_extractors function works correctly"""
    fe = get_available_pre_processing()
    assert "scale" in fe


def test_get_available_performance_indicators():
    """Test that get_available_performance_indicators function works correctly"""
    pi = get_available_performance_indicators()
    assert "accuracy" in pi


def test_get_available_unsupervised_methods():
    """Test that get_available_unsupervised_methods function works correctly"""
    cl = get_available_unsupervised_methods()
    # assert "kmeans" in cl


def test_get_available_supervised_methods():
    """Test that get_available_supervised_methods function works correctly"""
    cl = get_available_supervised_methods()
    # assert "logistic_regression" in cl


def test_get_available_semi_supervised_methods():
    """Test that get_available_semi_supervised_methods function works correctly"""
    cl = get_available_semi_supervised_methods()
    # assert "darr" in cl


def test_get_available_formatters():
    """Test that get_available_formatters function works correctly"""
    pi = get_available_formatters()
    assert "hdf5" in pi


def test_get_available_datafiles():
    """Test that get_available_datafiles function works correctly"""
    cl = get_available_datafiles()
    assert "audio" in cl


def test_get_available_pipeline_types():
    """Test that get_available_pipeline_types function works correctly"""
    cl = get_available_pipeline_types()
    assert "whale_detector" in cl


def test_get_available_datasets():
    """Test that get_available_datasets function works correctly"""
    ds = get_available_datasets()
    assert "files_fold" in ds


def test_load_parameters():
    """Test that loading set of parameters works the way it is expected"""
    p = Pipeline()

    demo_parameters = """
{
    "id": "demo_pipeline",
    "pipeline_type": "whale_detector",
    "input_data": [
        {
            "filename": "demo_data.h5",
            "datafile": "time_series",
            "formatter": "hdf5"
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
        "method": "kmeans"
    },
    "performance_indicators": [
        {
            "method": "accuracy",
            "parameters": {}
        }
    ]
}
    """

    p.load_parameters(demo_parameters)
    assert type(p.parameters["machine_learning"]) is dict
    assert p.parameters["performance_indicators"][0]["method"] == "accuracy"
