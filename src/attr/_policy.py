# SPDX-License-Identifier: MIT

"""
Thread-local policy stacks for the *attrs* runtime exits.

This module owns the assignment / evolution / export policy frames that let
middleware temporarily change what happens when, on the *current thread*:

- an attribute is set through an *attrs*-generated ``__setattr__``
  (``setattr_pipeline``),
- `attr.evolve` / ``__replace__`` runs (``evolve_surface``),
- `attr.assoc` runs (``assoc_surface``),
- `attr.asdict` / `attr.astuple` run (``export_surface``).

The stacks are deliberately *thread-local*: they are never a process-wide
singleton, two threads can hold different frames at the same time, and frames
do not leak across threads.  They never mutate class-level ``on_setattr`` or
frozen-ness; those permanent settings are merged with the current frame at the
respective exit instead.

An empty stack is equivalent to stock *attrs* behavior.
"""

from __future__ import annotations

import threading

from contextlib import contextmanager
from enum import Enum
from typing import Any, Callable, NamedTuple

from . import setters
from .exceptions import PolicyStackError


class SetattrPipeline(Enum):
    """
    What the generated ``__setattr__`` does with a single field write.
    """

    HONOR_HOOKS = "honor_hooks"
    """
    Run the field's permanent ``on_setattr`` pipeline exactly as configured
    on the class / attribute.
    """

    BARE_WRITE = "bare_write"
    """
    Store the value as given: do not run any converter, validator or other
    ``on_setattr`` hook.  Equivalent to an ``object.__setattr__`` write.
    """

    CONVERT_ONLY = "convert_only"
    """
    Run only the field's converter (if any); never its validator or other
    ``on_setattr`` hooks.
    """

    VALIDATE_ONLY = "validate_only"
    """
    Run only the field's validator (if any); never its converter or other
    ``on_setattr`` hooks.
    """


class EvolveSurface(Enum):
    """
    How `attr.evolve` / ``__replace__`` produce the replacement instance.
    """

    HONOR_REINIT = "honor_reinit"
    """
    Build the replacement through the class constructor, re-running the whole
    construction flight (including converters and validators).
    """

    DENY_EVOLVE = "deny_evolve"
    """
    Fail without producing a new instance.
    """

    PATCH_SETATTR = "patch_setattr"
    """
    Copy the instance and write the changed fields through the assignment
    pipeline instead of re-flying the constructor.  Converters of unchanged
    fields do not run again.
    """


class AssocSurface(Enum):
    """
    How the deprecated `attr.assoc` produces the copy.
    """

    HONOR_ASSOC = "honor_assoc"
    """
    Shallow-copy and bypass the constructor, writing changes by storage name
    through ``object.__setattr__``.
    """

    DENY_ASSOC = "deny_assoc"
    """
    Fail without producing a new instance.
    """

    FORCE_REINIT = "force_reinit"
    """
    Re-fly through the class constructor with the same reinitialization
    semantics as `attr.evolve`'s ``honor_reinit`` (converters run again).
    """


class ExportSurface(Enum):
    """
    How `attr.asdict` / `attr.astuple` honor the caller-supplied filter and
    collection handling for one export.
    """

    HONOR_FILTER = "honor_filter"
    """
    Honor the caller's *filter* and *retain_collection_types* arguments.
    """

    FORCE_INCLUDE = "force_include"
    """
    Ignore ``filters.exclude``-style filters for this export, so fields that
    the caller's exclude filter would drop are exported anyway.
    """

    DROP_COLLECTIONS_TO_BUILTIN = "drop_collections_to_builtin"
    """
    Convert tuple/set/frozenset containers to `list` and dict-valued
    containers to `dict`, even when *retain_collection_types* is true.
    """


class EffectivePolicy(NamedTuple):
    """
    Immutable, read-only view of the policy effective for the current thread.

    It is the merge of the innermost pushed frame with the permanent
    per-attribute / per-class settings at the consuming exit.  With no frame
    pushed, every dimension is its ``HONOR_*`` default.
    """

    setattr_pipeline: SetattrPipeline
    evolve_surface: EvolveSurface
    assoc_surface: AssocSurface
    export_surface: ExportSurface


_DEFAULT_POLICY = EffectivePolicy(
    SetattrPipeline.HONOR_HOOKS,
    EvolveSurface.HONOR_REINIT,
    AssocSurface.HONOR_ASSOC,
    ExportSurface.HONOR_FILTER,
)


class _PolicyFrame:
    """
    One immutable frame on a thread's stack.

    Frames are identity-compared by the exits to detect that the stack was
    changed while the exit was running.
    """

    __slots__ = (
        "assoc_surface",
        "evolve_surface",
        "export_surface",
        "setattr_pipeline",
    )

    def __init__(
        self,
        setattr_pipeline: SetattrPipeline,
        evolve_surface: EvolveSurface,
        assoc_surface: AssocSurface,
        export_surface: ExportSurface,
    ) -> None:
        self.setattr_pipeline = setattr_pipeline
        self.evolve_surface = evolve_surface
        self.assoc_surface = assoc_surface
        self.export_surface = export_surface


class _PolicyStackLocal(threading.local):
    def __init__(self) -> None:
        # One list *per thread*; this constructor runs once per thread.
        self.frames: list[_PolicyFrame] = []


_local = _PolicyStackLocal()


def _top() -> _PolicyFrame | None:
    """
    Return the innermost frame of the current thread's stack, or None.
    """
    frames = _local.frames
    return frames[-1] if frames else None


def current_policy() -> EffectivePolicy:
    """
    Return the read-only effective policy on the current thread.

    With no frame pushed, every dimension is its ``HONOR_*`` default, which is
    exactly stock *attrs* behavior.
    """
    frame = _top()
    if frame is None:
        return _DEFAULT_POLICY

    return EffectivePolicy(
        frame.setattr_pipeline,
        frame.evolve_surface,
        frame.assoc_surface,
        frame.export_surface,
    )


def _check_enum(value: Any, enum_cls: type[Enum], name: str) -> None:
    if value is not None and not isinstance(value, enum_cls):
        msg = f"{name} must be a member of {enum_cls.__name__}, got {value!r}."
        raise TypeError(msg)


def push_policy(
    *,
    setattr_pipeline: SetattrPipeline | None = None,
    evolve_surface: EvolveSurface | None = None,
    assoc_surface: AssocSurface | None = None,
    export_surface: ExportSurface | None = None,
) -> _PolicyFrame:
    """
    Push a policy frame onto the *current thread's* stack and return its
    opaque token.

    Dimensions left as None inherit the value currently in effect, so an
    inner frame only has to name what it changes.  Pair every push with
    `pop_policy` (or use `policy_frame`) so frames do not leak.
    """
    _check_enum(setattr_pipeline, SetattrPipeline, "setattr_pipeline")
    _check_enum(evolve_surface, EvolveSurface, "evolve_surface")
    _check_enum(assoc_surface, AssocSurface, "assoc_surface")
    _check_enum(export_surface, ExportSurface, "export_surface")

    current = current_policy()
    frame = _PolicyFrame(
        setattr_pipeline
        if setattr_pipeline is not None
        else current.setattr_pipeline,
        evolve_surface
        if evolve_surface is not None
        else current.evolve_surface,
        assoc_surface if assoc_surface is not None else current.assoc_surface,
        export_surface
        if export_surface is not None
        else current.export_surface,
    )
    _local.frames.append(frame)
    return frame


def pop_policy(token: _PolicyFrame | None = None) -> None:
    """
    Pop the current thread's innermost frame.

    Args:
        token: If given, it must be the frame returned by the matching
            `push_policy` call and still be the innermost frame.

    Raises:
        attrs.exceptions.PolicyStackError:
            If the stack is empty or *token* is not the innermost frame.
    """
    frames = _local.frames
    if not frames:
        msg = "Cannot pop a policy frame: the current thread's stack is empty."
        raise PolicyStackError(msg)

    if token is not None and token is not frames[-1]:
        msg = (
            "Cannot pop policy frame: the given token is not the innermost "
            "frame of the current thread's stack."
        )
        raise PolicyStackError(msg)

    frames.pop()


@contextmanager
def policy_frame(
    *,
    setattr_pipeline: SetattrPipeline | None = None,
    evolve_surface: EvolveSurface | None = None,
    assoc_surface: AssocSurface | None = None,
    export_surface: ExportSurface | None = None,
):
    """
    Context manager that pushes a frame on entry and always pops it again on
    exit, including when the body raises.
    """
    token = push_policy(
        setattr_pipeline=setattr_pipeline,
        evolve_surface=evolve_surface,
        assoc_surface=assoc_surface,
        export_surface=export_surface,
    )
    try:
        yield current_policy()
    finally:
        pop_policy(token)


def _ensure_unchanged(
    frame: _PolicyFrame | None, where: str, _restore=None
) -> None:
    """
    Reject an exit whose policy frame was changed (by an unbalanced push/pop)
    after the exit had started, instead of silently keeping an entry
    snapshot.  Balanced push/pop pairs inside user hooks restore *frame* and
    are allowed.

    *where* names the exit for the error message.  When *_restore* is given
    as ``(instance, name, previous_value)`` and the change is only discovered
    after storing (generator ``on_setattr`` hooks run code after the store),
    the previous value is put back before raising.
    """
    if _top() is not frame:
        if _restore is not None:
            inst, attr_name, old_value = _restore
            object.__setattr__(inst, attr_name, old_value)
        msg = (
            f"The {where} policy frame changed while {where} was already "
            "running. The in-flight operation is rejected; balance "
            "push_policy/pop_policy inside hooks or use policy_frame()."
        )
        raise PolicyStackError(msg)


# ---------------------------------------------------------------------------
# setattr exit
# ---------------------------------------------------------------------------


def dispatch_generated_setattr(self, name, val, entry) -> None:
    """
    Run the body of an *attrs*-generated ``__setattr__`` for a tracked field.

    *entry* is the ``(attribute, permanent_hook, is_generator)`` triple built
    once per class.  The effective ``setattr_pipeline`` is read live and
    re-checked after user code (hooks / converters / validators) ran, so a
    frame pushed mid-flight is rejected rather than ignored.
    """
    a, hook, is_gen = entry
    frame = _top()
    mode = (
        SetattrPipeline.HONOR_HOOKS
        if frame is None
        else frame.setattr_pipeline
    )

    if mode is SetattrPipeline.HONOR_HOOKS:
        if is_gen():
            gen = hook(self, a, val)
            nval = next(gen)
            # A frame pushed inside the pre-assignment part invalidates the
            # write before the value is stored.
            _ensure_unchanged(frame, "setattr")
            old_value = getattr(self, name)
            setattr_ = object.__setattr__
            setattr_(self, name, nval)
            try:
                next(gen)
            except StopIteration:
                _ensure_unchanged(
                    frame, "setattr", _restore=(self, name, old_value)
                )
                return

            gen.close()
            msg = "Generator on_setattr hook yielded more than once."
            raise RuntimeError(msg)

        nval = hook(self, a, val)
        _ensure_unchanged(frame, "setattr")
        object.__setattr__(self, name, nval)
        return

    if mode is SetattrPipeline.BARE_WRITE:
        nval = val
    elif mode is SetattrPipeline.CONVERT_ONLY:
        nval = setters.convert(self, a, val)
    elif mode is SetattrPipeline.VALIDATE_ONLY:
        nval = setters.validate(self, a, val)
    else:  # pragma: no cover
        msg = f"Unknown setattr pipeline policy: {mode!r}."
        raise PolicyStackError(msg)

    # The converter/validator may have pushed a frame: reject before storing.
    _ensure_unchanged(frame, "setattr")
    object.__setattr__(self, name, nval)


# ---------------------------------------------------------------------------
# export exit
# ---------------------------------------------------------------------------


_EXCLUDE_FILTER_MARKER = "__attrs_exclude_filter__"


def _is_exclude_filter(filter: Callable | None) -> bool:
    return callable(filter) and getattr(filter, _EXCLUDE_FILTER_MARKER, False)


def resolve_export_arguments(filter, retain_collection_types):
    """
    Merge the current thread's ``export_surface`` frame with the caller's
    *filter* / *retain_collection_types* arguments.

    Returns ``(frame, effective_filter, effective_retain)`` where *frame* is
    the entry-time top frame (or None) for end-of-exit validation.
    """
    frame = _top()
    if frame is None:
        return None, filter, retain_collection_types

    surface = frame.export_surface
    if surface is ExportSurface.FORCE_INCLUDE:
        if _is_exclude_filter(filter):
            filter = None
    elif surface is ExportSurface.DROP_COLLECTIONS_TO_BUILTIN:
        retain_collection_types = False

    return frame, filter, retain_collection_types
