# SPDX-License-Identifier: MIT

from attr import (
    NOTHING,
    AssocSurface,
    Attribute,
    AttrsInstance,
    Converter,
    EffectivePolicy,
    EvolveSurface,
    ExportSurface,
    Factory,
    NothingType,
    SetattrPipeline,
    _make_getattr,
    assoc,
    cmp_using,
    current_policy,
    define,
    evolve,
    field,
    fields,
    fields_dict,
    frozen,
    has,
    make_class,
    mutable,
    policy_frame,
    pop_policy,
    push_policy,
    resolve_types,
    validate,
)
from attr._make import ClassProps
from attr._next_gen import asdict, astuple, inspect

from . import exceptions, filters, setters


__all__ = [
    "NOTHING",
    "AssocSurface",
    "Attribute",
    "AttrsInstance",
    "ClassProps",
    "Converter",
    "EffectivePolicy",
    "EvolveSurface",
    "ExportSurface",
    "Factory",
    "NothingType",
    "SetattrPipeline",
    "__author__",
    "__copyright__",
    "__description__",
    "__doc__",
    "__email__",
    "__license__",
    "__title__",
    "__url__",
    "__version__",
    "__version_info__",
    "asdict",
    "assoc",
    "astuple",
    "cmp_using",
    "converters",
    "current_policy",
    "define",
    "evolve",
    "exceptions",
    "field",
    "fields",
    "fields_dict",
    "filters",
    "frozen",
    "has",
    "inspect",
    "make_class",
    "mutable",
    "policy_frame",
    "pop_policy",
    "push_policy",
    "resolve_types",
    "setters",
    "validate",
    "validators",
]

__getattr__ = _make_getattr(__name__)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
