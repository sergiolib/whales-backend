import json
from numpy import ndarray, array
from pandas import Series


class WhalesEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ndarray):
            return {"type": "ndarray", "data": obj.tolist()}

        if isinstance(obj, Series):
            return {"type": "series", "data": obj.tolist(), "index": obj.index.tolist()}

        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)


class WhalesDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        if isinstance(obj, dict):
            if "type" in obj:
                if obj["type"] == "ndarray":
                    return array(obj["data"])

                if obj["type"] == "series":
                    return Series(obj["data"], index=obj["index"])

        return obj
