import numpy as np

from whales.modules.features_extractors.mfcc import PipelineMethod as MFCC
from whales.modules.features_extractors.identity import PipelineMethod as Identity
from whales.modules.features_extractors.zero_crossing_rate import PipelineMethod as ZeroCrossingRate
from whales.modules.features_extractors.min import PipelineMethod as Min
from whales.modules.features_extractors.range import PipelineMethod as Range
from whales.modules.features_extractors.skewness import PipelineMethod as Skewness


def generate_data(n, d):
    return np.random.rand(n, d)


def test_mfcc():
    data = generate_data(10000, 2500)
    f = MFCC()
    t = f.transform(data=data)
    assert t.shape[0] == data.shape[0]
    assert f.description != ""
    assert t.ndim == 2


def test_identity():
    data = generate_data(10000, 2500)
    f = Identity()
    t = f.transform(data=data)
    np.testing.assert_allclose(data, t)
    assert f.description != ""
    assert t.ndim == 2


def test_zero_crossing_rate():
    data = generate_data(10000, 2500) - 0.5
    f = ZeroCrossingRate()
    t = f.transform(data=data)
    assert t.shape[0] == data.shape[0]
    assert t.shape[1] == 1
    assert f.description != ""
    assert t.ndim == 2


def test_min():
    data = generate_data(10000, 2500) - 0.5
    f = Min()
    t = f.transform(data=data)
    assert t.shape[0] == data.shape[0]
    assert t.shape[1] == 1
    assert f.description != ""
    assert t.ndim == 2


def test_range():
    data = generate_data(10000, 2500) - 0.5
    f = Range()
    t = f.transform(data=data)
    assert t.shape[0] == data.shape[0]
    assert t.shape[1] == 1
    assert f.description != ""
    assert t.ndim == 2


def test_skewness():
    data = generate_data(10000, 2500) - 0.5
    f = Skewness()
    t = f.transform(data=data)
    assert t.shape[0] == data.shape[0]
    assert t.shape[1] == 1
    assert f.description != ""
    assert t.ndim == 2
