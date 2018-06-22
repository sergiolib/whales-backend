from whales.modules.supervised.svm import SVM


def test_set_parameter():
    module = SVM()
    start_params = module.parameters.copy()
    module.parameters = {
        "new_parameter": 1234
    }
    end_params = module.parameters.copy()
    assert end_params == {**start_params, "new_parameter": 1234}
