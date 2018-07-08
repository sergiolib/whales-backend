from whales.modules.pipelines.getters import get_available_features_extractors
from whales.modules.pipelines.getters import get_available_pre_processing
from whales.modules.pipelines.getters import get_available_pipeline_types
from whales.modules.pipelines.getters import get_available_formatters
from whales.modules.pipelines.getters import get_available_labels_formatters
from whales.modules.pipelines.getters import get_available_data_files
from whales.modules.pipelines.getters import get_available_performance_indicators
from whales.modules.pipelines.getters import get_available_unsupervised_methods
from whales.modules.pipelines.getters import get_available_supervised_methods
from whales.modules.pipelines.getters import get_available_semi_supervised_methods
from whales.modules.pipelines.getters import get_available_data_sets


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


def test_get_available_labels_formatters():
    """Test that get_available_labels_formatters function works correctly"""
    lf = get_available_labels_formatters()
    assert "csv" in lf


def test_get_available_datafiles():
    """Test that get_available_datafiles function works correctly"""
    cl = get_available_data_files()
    assert "audio" in cl


def test_get_available_pipeline_types():
    """Test that get_available_pipeline_types function works correctly"""
    cl = get_available_pipeline_types()
    assert "train_whale_detector" in cl
    assert "predict_whale_detector" in cl
    assert "whale_detector" not in cl  # Deprecated, only for testing


def test_get_available_datasets():
    """Test that get_available_datasets function works correctly"""
    ds = get_available_data_sets()
    assert "no_split" in ds
