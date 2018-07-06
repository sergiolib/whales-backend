from numpy import array, ndarray
import json

from whales.utilities.json import WhalesEncoder, WhalesDecoder


def test_json_encode_decode_ndarray():
    a = array([[1, 2, 3, 4, 5]])
    json.dump(a, open("tmp.json", "w"), cls=WhalesEncoder)
    b = json.load(open("tmp.json", "r"), cls=WhalesDecoder)
    assert type(b) is ndarray
    assert (a == b).all()
