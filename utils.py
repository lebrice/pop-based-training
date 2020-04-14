from typing import TypeVar, Dict
import dataclasses
from dataclasses import Field

Dataclass = TypeVar("Dataclass")
T = TypeVar("T")

def field_dict(dataclass: Dataclass) -> Dict[str, Field]:
    result: Dict[str, Field] = {}
    for field in dataclasses.fields(dataclass):
        result[field.name] = field
    return result