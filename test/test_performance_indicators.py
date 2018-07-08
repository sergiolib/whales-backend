from os.path import isfile
import numpy as np
import matplotlib.pyplot as plt

from whales.modules.performance_indicators.accuracy import Accuracy
from whales.modules.performance_indicators.confusion_matrix import ConfusionMatrix


def test_accuracy():
    true = [1, 0, 0, 1]
    predicted = [0, 0, 0, 1]
    pi = Accuracy()
    pi.parameters = {
        "target": true,
        "prediction": predicted
    }
    result = pi.compute()
    assert result == 0.75
    pi.save_results("./tmp")
    assert isfile("./tmp.txt")
    with open("./tmp.txt", 'r') as res:
        res = res.readline()
        assert res == "0.75"


def test_confusion_matrix():
    true = [1, 0, 0, 1]
    predicted = [0, 0, 0, 1]
    pi = ConfusionMatrix()
    pi.parameters = {
        "target": true,
        "prediction": predicted,
    }
    result = pi.compute()
    assert result.tolist() == [[2, 0], [1, 1]]
    pi.save_results("./tmp")
    assert isfile("./tmp.txt")
    cf = np.loadtxt("./tmp.txt")
    np.testing.assert_allclose(cf, [[2, 0], [1, 1]])

    # Normalized
    pi.parameters["normalized"] = True
    result2 = pi.compute()
    assert result2.tolist() == [[0.5, 0.0], [0.25, 0.25]]

    # Plot
    pi.parameters["plot"] = True
    result3 = pi.compute()
    assert type(result3) is plt.Figure
    pi.save_results("./tmp")
    assert isfile("./tmp.png")

