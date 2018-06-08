from whales.modules.pipeline.pipeline import get_available_feature_extractors
from whales.modules.pipeline.pipeline import get_available_formatters
from whales.modules.pipeline.pipeline import get_available_datafiles
from whales.modules.pipeline.pipeline import get_available_performance_indicators
from whales.modules.pipeline.pipeline import get_available_clustering_methods


def test_get_available_feature_extractors():
    """Test that get_available_feature_extractors function works correctly"""
    fe = get_available_feature_extractors()
    assert "identity" in fe


def test_get_available_performance_indicators():
    """Test that get_available_performance_indicators function works correctly"""
    pi = get_available_performance_indicators()
    assert "accuracy" in pi


def test_get_available_clustering_methods():
    """Test that get_available_clustering_methods function works correctly"""
    cl = get_available_clustering_methods()
    assert "kmeans" in cl


def test_get_available_formatters():
    """Test that get_available_formatters function works correctly"""
    pi = get_available_formatters()
    assert "hdf5" in pi


def test_get_available_datafiles():
    """Test that get_available_datafiles function works correctly"""
    cl = get_available_datafiles()
    assert "time_series" in cl

