from whales.modules.pipeline.pipeline import get_available_feature_extractors
from whales.modules.pipeline.pipeline import get_available_performance_indicators


def test_get_available_feature_extractors():
    """Test that get_available_feature_extractors function works correctly"""
    fe = get_available_feature_extractors()
    assert "identity" in fe


def test_get_available_performance_indicators():
    """Test that get_available_performance_indicators function works correctly"""
    pi = get_available_performance_indicators()
    assert "accuracy" in pi
