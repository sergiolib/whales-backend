import numpy as np

from whales.modules.features_extractors.mfcc import PipelineMethod as MFCC
from whales.modules.features_extractors.identity import PipelineMethod as Identity
from whales.modules.features_extractors.zero_crossing_rate import PipelineMethod as ZeroCrossingRate
from whales.modules.features_extractors.min import PipelineMethod as Min


def generate_data(n, d):
    return np.random.rand(n, d)


def test_mfcc():
    data = generate_data(10000, 2500)
    f = MFCC()
    t = f.transform(data=data)
    assert t.shape[0] == data.shape[0]
    assert f.description != ""


def test_identity():
    data = generate_data(10000, 2500)
    f = Identity()
    t = f.transform(data=data)
    np.testing.assert_allclose(data, t)
    assert f.description != ""


def test_zero_crossing_rate():
    data = generate_data(10000, 2500) - 0.5
    f = ZeroCrossingRate()
    t = f.transform(data=data)
    assert f.description != ""


def test_min():
    data = generate_data(10000, 2500) - 0.5
    f = Min()
    t = f.transform(data=data)
    assert f.description != ""
