from whales.modules.pipeline.pipeline import get_available_feature_extractors


def test_get_available_feature_extractors():
    """Test that get_available_feature_extractors function works correctly"""
    fe = get_available_feature_extractors()
    assert "identity" in fe
