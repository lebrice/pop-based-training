import dataclasses
from contextlib import contextmanager
from dataclasses import Field
from typing import Dict, TypeVar

Dataclass = TypeVar("Dataclass")
T = TypeVar("T")

def field_dict(dataclass: Dataclass) -> Dict[str, Field]:
    result: Dict[str, Field] = {}
    for field in dataclasses.fields(dataclass):
        result[field.name] = field
    return result

@contextmanager
def requires_import(package):
    try:
        pkg = __import__(package)
        yield pkg
    except ImportError as e:
        yield None
    finally:
        return


if __name__ == "__main__":
    class Bob:
        @requires_import("numpy")
        def do_something_requiring_numpy(self):
            import numpy as np
            return np.ones(1)

        @requires_import("numpy")
        @requires_import("torch")
        def do_something_requiring_numpy_and_torch(self):
            import numpy as np
            import torch
            print(np.zeros(1))
            print(torch.zeros(1))
            return np.ones(1)

    b = Bob()
    print("result: (numpy)", b.do_something_requiring_numpy())
    print("result: (pytorch)", b.do_something_requiring_numpy_and_torch())

    with requires_import("torch") as torch:
        print("bob:", torch.zeros(1))
        print("HJEY", np.zeros(1))
        print("HEHE")
    print("HOHO")
    exit()
