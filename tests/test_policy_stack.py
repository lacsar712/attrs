# SPDX-License-Identifier: MIT

"""
Acceptance tests for the thread-local setattr / evolve / assoc / export
policy stacks.

The four dimensions are independent enums; frames are thread-local and never
process-global.  An empty stack must be indistinguishable from stock attrs
behavior.
"""

import threading

import pytest

import attr
import attrs

from attr import setters, validators
from attr._compat import PY_3_13_PLUS
from attr.exceptions import (
    FrozenAttributeError,
    FrozenInstanceError,
    PolicyStackError,
)


SetattrPipeline = attr.SetattrPipeline
EvolveSurface = attr.EvolveSurface
AssocSurface = attr.AssocSurface
ExportSurface = attr.ExportSurface


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


class Counter:
    """Counts converter / validator invocations per field label."""

    def __init__(self):
        self.events = []

    def converter(self, label):
        def conv(value):
            self.events.append(("convert", label))
            return int(value)

        return conv

    def validator(self, label):
        def validate(inst, attribute, value):
            self.events.append(("validate", label))
            if not isinstance(value, int):
                msg = f"{label} must be int, got {value!r}"
                raise TypeError(msg)

        return validate

    def count(self, kind, label):
        return self.events.count((kind, label))


def make_hook_class(slots):
    """
    Non-frozen class whose post-init writes run the convert+validate pipe.
    Built with the classic ``attr.s`` API.
    """
    counter = Counter()

    @attr.s(slots=slots, on_setattr=[setters.convert, setters.validate])
    class Hooked:
        x = attr.ib(
            converter=counter.converter("x"),
            validator=counter.validator("x"),
        )
        y = attr.ib(
            converter=counter.converter("y"),
            validator=counter.validator("y"),
        )

    return Hooked, counter


# --------------------------------------------------------------------------
# empty stack / read-only effective view / stack mechanics
# --------------------------------------------------------------------------


class TestEmptyStackAndView:
    def test_default_view_is_all_honor(self):
        policy = attr.current_policy()
        assert isinstance(policy, attr.EffectivePolicy)
        assert policy.setattr_pipeline is SetattrPipeline.HONOR_HOOKS
        assert policy.evolve_surface is EvolveSurface.HONOR_REINIT
        assert policy.assoc_surface is AssocSurface.HONOR_ASSOC
        assert policy.export_surface is ExportSurface.HONOR_FILTER

    def test_view_is_read_only(self):
        policy = attr.current_policy()
        with pytest.raises(AttributeError):
            policy.setattr_pipeline = SetattrPipeline.BARE_WRITE

    def test_dimensions_are_independent_enums(self):
        enums = (
            SetattrPipeline,
            EvolveSurface,
            AssocSurface,
            ExportSurface,
        )
        assert len({id(e) for e in enums}) == 4
        for enum in enums:
            assert not isinstance(True, enum)
        assert {m.name for m in SetattrPipeline} == {
            "HONOR_HOOKS",
            "BARE_WRITE",
            "CONVERT_ONLY",
            "VALIDATE_ONLY",
        }
        assert {m.name for m in EvolveSurface} == {
            "HONOR_REINIT",
            "DENY_EVOLVE",
            "PATCH_SETATTR",
        }
        assert {m.name for m in AssocSurface} == {
            "HONOR_ASSOC",
            "DENY_ASSOC",
            "FORCE_REINIT",
        }
        assert {m.name for m in ExportSurface} == {
            "HONOR_FILTER",
            "FORCE_INCLUDE",
            "DROP_COLLECTIONS_TO_BUILTIN",
        }

    def test_nested_frames_inherit_unspecified_dimensions(self):
        with attr.policy_frame(
            setattr_pipeline=SetattrPipeline.BARE_WRITE
        ) as outer:
            assert outer.setattr_pipeline is SetattrPipeline.BARE_WRITE
            assert outer.evolve_surface is EvolveSurface.HONOR_REINIT

            with attr.policy_frame(
                evolve_surface=EvolveSurface.DENY_EVOLVE
            ) as inner:
                assert inner.setattr_pipeline is SetattrPipeline.BARE_WRITE
                assert inner.evolve_surface is EvolveSurface.DENY_EVOLVE

            # inner pop exposes outer frame again
            back = attr.current_policy()
            assert back.setattr_pipeline is SetattrPipeline.BARE_WRITE
            assert back.evolve_surface is EvolveSurface.HONOR_REINIT

        assert attr.current_policy() == (
            SetattrPipeline.HONOR_HOOKS,
            EvolveSurface.HONOR_REINIT,
            AssocSurface.HONOR_ASSOC,
            ExportSurface.HONOR_FILTER,
        )

    def test_context_manager_pops_on_exception(self):
        with (
            attr.policy_frame(assoc_surface=AssocSurface.DENY_ASSOC),
            pytest.raises(ValueError, match="boom"),
        ):
            raise ValueError("boom")

        assert attr.current_policy().assoc_surface is AssocSurface.HONOR_ASSOC

    def test_pop_on_empty_stack_raises(self):
        with pytest.raises(PolicyStackError):
            attr.pop_policy()

    def test_pop_with_foreign_token_raises(self):
        token = attr.push_policy(setattr_pipeline=SetattrPipeline.BARE_WRITE)
        other = attr.push_policy(setattr_pipeline=SetattrPipeline.CONVERT_ONLY)
        with pytest.raises(PolicyStackError):
            attr.pop_policy(token)
        attr.pop_policy(other)
        attr.pop_policy(token)
        assert (
            attr.current_policy().setattr_pipeline
            is SetattrPipeline.HONOR_HOOKS
        )

    def test_push_rejects_wrong_enum_type(self):
        with pytest.raises(TypeError):
            attr.push_policy(setattr_pipeline=EvolveSurface.DENY_EVOLVE)
        with pytest.raises(TypeError):
            attr.push_policy(export_surface=True)
        assert attr.current_policy().export_surface is (
            ExportSurface.HONOR_FILTER
        )

    @pytest.mark.parametrize("slots", [True, False])
    def test_empty_stack_behavior_unchanged(self, slots):
        Hooked, counter = make_hook_class(slots)
        inst = Hooked("1", "2")
        counter.events.clear()

        inst.y = "5"
        assert inst.y == 5
        assert counter.events == [("convert", "y"), ("validate", "y")]

        # honor path: the converter fails on non-numeric input first
        with pytest.raises((TypeError, ValueError)):
            inst.y = "not-an-int"
        assert inst.y == 5


# --------------------------------------------------------------------------
# 1) setattr_pipeline
# --------------------------------------------------------------------------


class TestSetattrPipeline:
    @pytest.mark.parametrize("slots", [True, False])
    def test_honor_hooks_runs_convert_and_validate(self, slots):
        Hooked, counter = make_hook_class(slots)
        inst = Hooked("1", "2")
        counter.events.clear()

        with attr.policy_frame(setattr_pipeline=SetattrPipeline.HONOR_HOOKS):
            inst.x = "42"

        assert inst.x == 42
        assert counter.events == [("convert", "x"), ("validate", "x")]

    @pytest.mark.parametrize("slots", [True, False])
    def test_bare_write_stores_raw_final_value(self, slots):
        Hooked, counter = make_hook_class(slots)
        inst = Hooked("1", "2")
        counter.events.clear()

        with attr.policy_frame(setattr_pipeline=SetattrPipeline.BARE_WRITE):
            inst.x = "42"  # not converted, not validated

        assert inst.x == "42"
        assert counter.events == []

        # stock behavior returns after the frame is popped
        inst.x = "7"
        assert inst.x == 7
        assert counter.events == [("convert", "x"), ("validate", "x")]

    @pytest.mark.parametrize("slots", [True, False])
    def test_convert_only_skips_validation(self, slots):
        Hooked, counter = make_hook_class(slots)
        inst = Hooked("1", "2")
        counter.events.clear()

        with attr.policy_frame(setattr_pipeline=SetattrPipeline.CONVERT_ONLY):
            inst.x = "42"

        assert inst.x == 42
        assert counter.events == [("convert", "x")]

    @pytest.mark.parametrize("slots", [True, False])
    def test_validate_only_skips_conversion(self, slots):
        Hooked, counter = make_hook_class(slots)
        inst = Hooked("1", "2")
        counter.events.clear()

        with attr.policy_frame(setattr_pipeline=SetattrPipeline.VALIDATE_ONLY):
            inst.x = 42  # already int: validator passes, converter skipped

        assert inst.x == 42
        assert counter.events == [("validate", "x")]

        with (
            attr.policy_frame(setattr_pipeline=SetattrPipeline.VALIDATE_ONLY),
            pytest.raises(TypeError),
        ):
            inst.x = "not-an-int"  # validated raw: no conversion rescue
        assert inst.x == 42

    def test_define_s_make_class_consistent(self):
        counter = Counter()

        @attrs.define
        class Defined:
            x: int = attrs.field(
                converter=counter.converter("x"),
                validator=counter.validator("x"),
            )

        Made = attr.make_class(
            "Made",
            {
                "x": attr.ib(
                    converter=counter.converter("x"),
                    validator=counter.validator("x"),
                    on_setattr=[setters.convert, setters.validate],
                )
            },
            slots=True,
        )

        for cls in (Defined, Made):
            counter.events.clear()
            inst = cls("1")
            counter.events.clear()
            with attr.policy_frame(
                setattr_pipeline=SetattrPipeline.BARE_WRITE
            ):
                inst.x = "raw"
            assert inst.x == "raw"
            assert counter.events == []

    def test_bare_write_bypasses_field_level_frozen_hook(self):
        @attr.s
        class FieldFrozen:
            x = attr.ib(default=1, on_setattr=setters.frozen)

        inst = FieldFrozen()
        with pytest.raises(FrozenAttributeError):
            inst.x = 2

        with attr.policy_frame(setattr_pipeline=SetattrPipeline.BARE_WRITE):
            inst.x = 2
        assert inst.x == 2

    def test_stack_does_not_unfreeze_frozen_class(self):
        @attrs.frozen
        class Frozen:
            x: int

        inst = Frozen(1)
        for mode in (
            SetattrPipeline.BARE_WRITE,
            SetattrPipeline.CONVERT_ONLY,
            SetattrPipeline.VALIDATE_ONLY,
        ):
            with (
                attr.policy_frame(setattr_pipeline=mode),
                pytest.raises(FrozenInstanceError),
            ):
                inst.x = 2

    def test_late_push_after_pipeline_started_is_rejected(self):
        def late_hook(inst, attribute, value):
            nval = setters.convert(inst, attribute, value)
            setters.validate(inst, attribute, nval)
            # The write is already inside the on_setattr pipeline: pushing
            # another frame now must invalidate the write, not be ignored.
            attr.push_policy(setattr_pipeline=SetattrPipeline.BARE_WRITE)
            return nval

        @attr.s(on_setattr=late_hook)
        class Late:
            x = attr.ib(converter=int, validator=validators.instance_of(int))

        inst = Late(1)
        with pytest.raises(PolicyStackError):
            inst.x = "42"
        # rejected before storage
        assert inst.x == 1
        attr.pop_policy()

    def test_balanced_push_pop_inside_hook_is_allowed(self):
        def balanced_hook(inst, attribute, value):
            with attr.policy_frame(
                setattr_pipeline=SetattrPipeline.BARE_WRITE
            ):
                pass
            nval = setters.convert(inst, attribute, value)
            setters.validate(inst, attribute, nval)
            return nval

        @attr.s(on_setattr=balanced_hook)
        class Balanced:
            x = attr.ib(converter=int, validator=validators.instance_of(int))

        inst = Balanced(1)
        inst.x = "42"
        assert inst.x == 42
        assert (
            attr.current_policy().setattr_pipeline
            is SetattrPipeline.HONOR_HOOKS
        )

    def test_generator_hook_late_push_restores_value(self):
        def gen_hook(inst, attribute, value):
            nval = setters.convert(inst, attribute, value)
            yield nval
            # post-store push must reject and roll the stored value back
            attr.push_policy(setattr_pipeline=SetattrPipeline.BARE_WRITE)

        @attr.s(on_setattr=gen_hook)
        class Gen:
            x = attr.ib(converter=int)

        inst = Gen(1)
        with pytest.raises(PolicyStackError):
            inst.x = "42"
        assert inst.x == 1
        attr.pop_policy()


# --------------------------------------------------------------------------
# 2) evolve_surface
# --------------------------------------------------------------------------


class TestEvolveSurface:
    def test_honor_reinit_reflies_every_converter(self):
        counter = Counter()

        @attrs.define
        class Evolvable:
            x: int = attrs.field(
                converter=counter.converter("x"),
                validator=counter.validator("x"),
            )
            y: int = attrs.field(
                converter=counter.converter("y"),
                validator=counter.validator("y"),
            )

        inst = Evolvable("1", "2")
        counter.events.clear()

        new = attr.evolve(inst, y="3")

        assert new is not inst
        assert (new.x, new.y) == (1, 3)
        # The whole constructor re-flew: even unchanged x is converted again.
        assert counter.count("convert", "x") == 1
        assert counter.count("convert", "y") == 1
        assert counter.count("validate", "x") == 1
        assert counter.count("validate", "y") == 1

    def test_patch_setattr_skips_unchanged_fields(self):
        counter = Counter()

        @attrs.define
        class Patchable:
            x: int = attrs.field(
                converter=counter.converter("x"),
                validator=counter.validator("x"),
            )
            y: int = attrs.field(
                converter=counter.converter("y"),
                validator=counter.validator("y"),
            )

        inst = Patchable("1", "2")
        counter.events.clear()

        with attr.policy_frame(evolve_surface=EvolveSurface.PATCH_SETATTR):
            new = attr.evolve(inst, y="3")

        assert new is not inst
        assert (new.x, new.y) == (1, 3)
        # Changed field went through the assignment pipe; x never re-flew.
        assert counter.count("convert", "x") == 0
        assert counter.count("validate", "x") == 0
        assert counter.count("convert", "y") == 1
        assert counter.count("validate", "y") == 1

    def test_patch_setattr_differs_from_honor_on_init_false_factory(self):
        @attrs.define
        class WithFactory:
            a: int
            extra: list = attrs.field(default=attr.Factory(list), init=False)

        inst = WithFactory(1)

        new_honor = attr.evolve(inst, a=2)
        assert new_honor.extra is not inst.extra  # factory re-flew

        with attr.policy_frame(evolve_surface=EvolveSurface.PATCH_SETATTR):
            new_patch = attr.evolve(inst, a=2)
        assert new_patch.extra is inst.extra  # constructor never ran

    def test_deny_evolve_fails_without_new_instance(self):
        @attrs.define
        class Denied:
            x: int

        inst = Denied(1)
        with (
            attr.policy_frame(evolve_surface=EvolveSurface.DENY_EVOLVE),
            pytest.raises(PolicyStackError),
        ):
            attr.evolve(inst, x=2)
        assert inst.x == 1

    def test_deny_evolve_independent_of_assoc(self):
        @attrs.define
        class Both:
            x: int

        inst = Both(1)
        with attr.policy_frame(evolve_surface=EvolveSurface.DENY_EVOLVE):
            with pytest.raises(PolicyStackError):
                attr.evolve(inst, x=2)
            # assoc keeps its own surface and still works
            copied = attr.assoc(inst, x=2)
        assert copied.x == 2

    def test_patch_composes_with_setattr_pipeline_frame(self):
        counter = Counter()

        @attrs.define
        class Composed:
            y: int = attrs.field(
                converter=counter.converter("y"),
                validator=counter.validator("y"),
            )

        inst = Composed("1")
        counter.events.clear()

        with attr.policy_frame(
            evolve_surface=EvolveSurface.PATCH_SETATTR,
            setattr_pipeline=SetattrPipeline.BARE_WRITE,
        ):
            new = attr.evolve(inst, y="3")

        # patch routed the change through __setattr__, which saw bare_write
        assert new.y == "3"
        assert counter.events == []

    def test_unknown_change_under_patch_raises_type_error(self):
        @attrs.define
        class Small:
            x: int

        inst = Small(1)
        with (
            attr.policy_frame(evolve_surface=EvolveSurface.PATCH_SETATTR),
            pytest.raises(TypeError),
        ):
            attr.evolve(inst, nope=2)

    def test_late_deny_push_during_construction_is_rejected(self):
        armed = {"on": False}

        def maybe_push(label):
            def conv(value):
                if armed["on"] and label == "y":
                    attr.push_policy(evolve_surface=EvolveSurface.DENY_EVOLVE)
                return int(value)

            return conv

        @attrs.define
        class Armed:
            x: int = attrs.field(converter=maybe_push("x"))
            y: int = attrs.field(converter=maybe_push("y"))

        inst = Armed("1", "2")
        armed["on"] = True
        with pytest.raises(PolicyStackError):
            attr.evolve(inst, y="3")
        attr.pop_policy()
        assert inst.y == 2

    @pytest.mark.skipif(
        not PY_3_13_PLUS, reason="__replace__ is only generated on 3.13+"
    )
    def test_replace_dunder_reads_stack(self):
        import copy

        @attrs.define
        class Replaceable:
            x: int
            y: int = 0

        inst = Replaceable(1)
        with (
            attr.policy_frame(evolve_surface=EvolveSurface.DENY_EVOLVE),
            pytest.raises(PolicyStackError),
        ):
            copy.replace(inst, x=2)


# --------------------------------------------------------------------------
# 3) assoc_surface
# --------------------------------------------------------------------------


class TestAssocSurface:
    def test_honor_assoc_bypasses_constructor_and_pipeline(self):
        counter = Counter()
        init_runs = []

        @attr.s(on_setattr=[setters.convert, setters.validate])
        class Assocable:
            x = attr.ib(
                converter=counter.converter("x"),
                validator=counter.validator("x"),
            )

            def __attrs_post_init__(self):
                init_runs.append(1)

        inst = Assocable("1")
        counter.events.clear()
        init_runs.clear()

        new = attr.assoc(inst, x="9")

        assert new is not inst
        assert new.x == "9"  # storage-name bypass: no conversion
        assert counter.events == []  # not in assigning flight, no hooks
        assert init_runs == []

    def test_honor_assoc_uses_storage_name_for_private_attrs(self):
        @attr.s
        class Private:
            _x = attr.ib()

        inst = Private(1)
        new = attr.assoc(inst, _x=2)
        assert new._x == 2

    def test_deny_assoc_fails_without_copy(self):
        @attrs.define
        class DeniedAssoc:
            x: int

        inst = DeniedAssoc(1)
        with (
            attr.policy_frame(assoc_surface=AssocSurface.DENY_ASSOC),
            pytest.raises(PolicyStackError),
        ):
            attr.assoc(inst, x=2)
        assert inst.x == 1

    def test_force_reinit_reflies_like_evolve(self):
        counter = Counter()

        @attrs.define
        class ForceReinit:
            x: int = attrs.field(converter=counter.converter("x"))
            y: int = attrs.field(converter=counter.converter("y"))

        inst = ForceReinit("1", "2")
        counter.events.clear()

        with attr.policy_frame(assoc_surface=AssocSurface.FORCE_REINIT):
            new = attr.assoc(inst, y="9")

        assert (new.x, new.y) == (1, 9)
        # constructor re-flew: both converters ran (honor assoc runs none)
        assert counter.count("convert", "x") == 1
        assert counter.count("convert", "y") == 1

    def test_assoc_surface_independent_from_evolve_surface(self):
        @attrs.define
        class Independent:
            x: int

        inst = Independent(1)

        # assoc denied but evolve still reinitializes
        with attr.policy_frame(assoc_surface=AssocSurface.DENY_ASSOC):
            with pytest.raises(PolicyStackError):
                attr.assoc(inst, x=2)
            evolved = attr.evolve(inst, x=2)
        assert evolved.x == 2

        # evolve denied but assoc honor still copies
        with attr.policy_frame(evolve_surface=EvolveSurface.DENY_EVOLVE):
            with pytest.raises(PolicyStackError):
                attr.evolve(inst, x=2)
            copied = attr.assoc(inst, x=2)
        assert copied.x == 2


# --------------------------------------------------------------------------
# 4) export_surface
# --------------------------------------------------------------------------


class TestExportSurface:
    @attrs.define
    class Bag:
        a: int
        b: int
        secret: int

    def test_asdict_force_include_ignores_exclude_filter(self):
        inst = self.Bag(1, 2, 3)
        exclude_secret = attrs.filters.exclude("secret")

        assert attr.asdict(inst, filter=exclude_secret) == {
            "a": 1,
            "b": 2,
        }

        with attr.policy_frame(export_surface=ExportSurface.FORCE_INCLUDE):
            assert attr.asdict(inst, filter=exclude_secret) == {
                "a": 1,
                "b": 2,
                "secret": 3,
            }

    def test_force_include_keeps_include_and_arbitrary_filters(self):
        inst = self.Bag(1, 2, 3)
        only_a = attrs.filters.include("a")

        with attr.policy_frame(export_surface=ExportSurface.FORCE_INCLUDE):
            assert attr.asdict(inst, filter=only_a) == {"a": 1}
            assert attr.asdict(inst, filter=lambda a, v: a.name != "b") == {
                "a": 1,
                "secret": 3,
            }

    def test_asdict_drop_collections_overrides_retain(self):
        @attrs.define
        class Holder:
            t: tuple
            s: set
            d: dict

        inst = Holder((1, 2), {3}, {"k": 5})

        retained = attr.asdict(inst, retain_collection_types=True)
        assert type(retained["t"]) is tuple
        assert type(retained["s"]) is set
        assert type(retained["d"]) is dict

        with attr.policy_frame(
            export_surface=ExportSurface.DROP_COLLECTIONS_TO_BUILTIN
        ):
            dropped = attr.asdict(inst, retain_collection_types=True)
        assert type(dropped["t"]) is list
        assert type(dropped["s"]) is list
        assert type(dropped["d"]) is dict

    def test_attrs_asdict_retain_default_overridden(self):
        @attrs.define
        class Holder:
            t: tuple = ()

        inst = Holder((1, 2))
        assert type(attrs.asdict(inst)["t"]) is tuple

        with attr.policy_frame(
            export_surface=ExportSurface.DROP_COLLECTIONS_TO_BUILTIN
        ):
            assert type(attrs.asdict(inst)["t"]) is list

    def test_astuple_force_include_ignores_exclude_filter(self):
        inst = self.Bag(1, 2, 3)
        exclude_secret = attrs.filters.exclude("secret")

        assert attr.astuple(inst, filter=exclude_secret) == (1, 2)

        with attr.policy_frame(export_surface=ExportSurface.FORCE_INCLUDE):
            assert attr.astuple(inst, filter=exclude_secret) == (1, 2, 3)

    def test_astuple_drop_collections_overrides_retain(self):
        @attrs.define
        class Holder:
            t: tuple
            s: set
            d: dict

        inst = Holder((1, 2), {3}, {"k": 5})

        retained = attr.astuple(inst, retain_collection_types=True)
        assert type(retained[0]) is tuple
        assert type(retained[1]) is set
        assert type(retained[2]) is dict

        with attr.policy_frame(
            export_surface=ExportSurface.DROP_COLLECTIONS_TO_BUILTIN
        ):
            dropped = attr.astuple(inst, retain_collection_types=True)
        assert type(dropped[0]) is list
        assert type(dropped[1]) is list
        assert type(dropped[2]) is dict

    def test_late_force_include_push_during_asdict_is_rejected(self):
        inst = self.Bag(1, 2, 3)

        def push_in_filter(attribute, value):
            if attribute.name == "b":
                attr.push_policy(export_surface=ExportSurface.FORCE_INCLUDE)
            return True

        with pytest.raises(PolicyStackError):
            attr.asdict(inst, filter=push_in_filter)
        attr.pop_policy()

    def test_late_drop_push_during_astuple_is_rejected(self):
        @attrs.define
        class Holder:
            a: int
            t: tuple = ()

        inst = Holder(1, (2, 3))

        def push_in_filter(attribute, value):
            if attribute.name == "t":
                attr.push_policy(
                    export_surface=ExportSurface.DROP_COLLECTIONS_TO_BUILTIN
                )
            return True

        with pytest.raises(PolicyStackError):
            attr.astuple(inst, filter=push_in_filter)
        attr.pop_policy()

    def test_balanced_push_inside_filter_is_allowed(self):
        inst = self.Bag(1, 2, 3)

        def balanced_filter(attribute, value):
            with attr.policy_frame(export_surface=ExportSurface.FORCE_INCLUDE):
                pass
            return attribute.name != "secret"

        assert attr.asdict(inst, filter=balanced_filter) == {
            "a": 1,
            "b": 2,
        }


# --------------------------------------------------------------------------
# cross-cutting: custom __init__ calling __attrs_init__
# --------------------------------------------------------------------------


class TestCustomInit:
    def _make(self):
        @attr.s(
            init=False,
            on_setattr=[setters.convert, setters.validate],
        )
        class CustomInit:
            x = attr.ib(converter=int, validator=validators.instance_of(int))

            def __init__(self, x):
                self.__attrs_init__(x)

        return CustomInit

    def test_post_init_assignment_reads_stack(self):
        CustomInit = self._make()
        inst = CustomInit("1")

        with attr.policy_frame(setattr_pipeline=SetattrPipeline.BARE_WRITE):
            inst.x = "raw"
        assert inst.x == "raw"

        inst.x = "5"
        assert inst.x == 5

    def test_evolve_assoc_asdict_read_stack(self):
        CustomInit = self._make()
        inst = CustomInit("1")

        new = attr.evolve(inst, x="2")
        assert new.x == 2

        with attr.policy_frame(evolve_surface=EvolveSurface.PATCH_SETATTR):
            patched = attr.evolve(inst, x="3")
        assert patched.x == 3

        with (
            attr.policy_frame(assoc_surface=AssocSurface.DENY_ASSOC),
            pytest.raises(PolicyStackError),
        ):
            attr.assoc(inst, x=4)
        assert attr.assoc(inst, x=4).x == 4

        with attr.policy_frame(export_surface=ExportSurface.FORCE_INCLUDE):
            exported = attr.asdict(patched, filter=attrs.filters.exclude("x"))
        assert exported == {"x": 3}


# --------------------------------------------------------------------------
# thread isolation
# --------------------------------------------------------------------------


class TestThreadIsolation:
    def test_setattr_pipeline_does_not_cross_threads(self):
        Hooked, _ = make_hook_class(slots=True)
        results = {}
        errors = []
        barrier = threading.Barrier(2)

        def bare_thread():
            try:
                with attr.policy_frame(
                    setattr_pipeline=SetattrPipeline.BARE_WRITE
                ):
                    inst = Hooked(1, 1)
                    barrier.wait()
                    inst.x = "9"
                    results["bare"] = (
                        inst.x,
                        attr.current_policy().setattr_pipeline,
                    )
            except Exception:  # noqa: BLE001  # pragma: no cover
                import traceback

                errors.append(traceback.format_exc())

        def honor_thread():
            try:
                inst = Hooked(1, 1)
                # no frame on this thread: the other thread's frame must not
                # leak over
                assert (
                    attr.current_policy().setattr_pipeline
                    is SetattrPipeline.HONOR_HOOKS
                )
                barrier.wait()
                inst.x = "9"
                results["honor"] = (
                    inst.x,
                    attr.current_policy().setattr_pipeline,
                )
            except Exception:  # noqa: BLE001  # pragma: no cover
                import traceback

                errors.append(traceback.format_exc())

        t1 = threading.Thread(target=bare_thread)
        t2 = threading.Thread(target=honor_thread)
        t1.start()
        t2.start()
        t1.join(5)
        t2.join(5)

        assert not errors
        assert not t1.is_alive()
        assert not t2.is_alive()
        assert results["bare"] == ("9", SetattrPipeline.BARE_WRITE)
        assert results["honor"] == (9, SetattrPipeline.HONOR_HOOKS)
        assert (
            attr.current_policy().setattr_pipeline
            is SetattrPipeline.HONOR_HOOKS
        )

    def test_export_surface_does_not_cross_threads(self):
        @attrs.define
        class Row:
            a: int
            secret: int

        inst = Row(1, 2)
        exclude = attrs.filters.exclude("secret")
        results = {}
        errors = []
        barrier = threading.Barrier(2)

        def force_thread():
            try:
                with attr.policy_frame(
                    export_surface=ExportSurface.FORCE_INCLUDE
                ):
                    barrier.wait()
                    results["force"] = attr.asdict(inst, filter=exclude)
            except Exception:  # noqa: BLE001  # pragma: no cover
                import traceback

                errors.append(traceback.format_exc())

        def honor_thread():
            try:
                assert (
                    attr.current_policy().export_surface
                    is ExportSurface.HONOR_FILTER
                )
                barrier.wait()
                results["honor"] = attr.asdict(inst, filter=exclude)
            except Exception:  # noqa: BLE001  # pragma: no cover
                import traceback

                errors.append(traceback.format_exc())

        t1 = threading.Thread(target=force_thread)
        t2 = threading.Thread(target=honor_thread)
        t1.start()
        t2.start()
        t1.join(5)
        t2.join(5)

        assert not errors
        assert results["force"] == {"a": 1, "secret": 2}
        assert results["honor"] == {"a": 1}
