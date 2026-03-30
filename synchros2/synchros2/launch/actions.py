# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
from __future__ import annotations

from enum import Enum
from typing import Any, Final, List, Type

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.utilities.type_utils import coerce_to_type

_BOOLEAN_STR_CHOICES: Final[List[str]] = ["true", "True", "false", "False"]
_BOOLEAN_CHOICES: Final[List[str | bool]] = [*_BOOLEAN_STR_CHOICES, True, False]
_OPTIONAL_CHOICES: Final[List[str]] = [""]


def convert_to_bool(param_name: str, val: str) -> bool:
    """Converts a ros parameter to a bool"""
    try:
        return coerce_to_type(val.lower(), bool)
    except ValueError:
        print(f"Cannot convert `{param_name}` to bool (value is {val})")
        raise


def update_sigterm_sigkill_timeout(
    ld: LaunchDescription,
    *,
    sigterm_timeout_s: float = 60,
    sigkill_timeout_s: float = 60,
) -> None:
    """Increases the timeout for launch to escalate to SIGTERM and SIGKILL after you CTRL+C"""
    ld.add_action(DeclareLaunchArgument("sigterm_timeout", default_value=str(sigterm_timeout_s)))
    ld.add_action(DeclareLaunchArgument("sigkill_timeout", default_value=str(sigkill_timeout_s)))


class DeclareBooleanLaunchArgument(DeclareLaunchArgument):
    """Thin wrapper on `DeclareLaunchArgument` to restrict the choices to boolean"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if "choices" in kwargs:
            raise KeyError("Cannot set `choices` for `DeclareBooleanLaunchArgument`")
        if "default_value" in kwargs:
            default_value = kwargs["default_value"]
            if default_value not in _BOOLEAN_CHOICES:
                raise ValueError(f"`default_value` must be from {_BOOLEAN_CHOICES}")
            if isinstance(default_value, bool):
                kwargs["default_value"] = "true" if default_value else "false"

        super().__init__(*args, choices=_BOOLEAN_STR_CHOICES, **kwargs)


class DeclareEnumLaunchArgument(DeclareLaunchArgument):
    """Thin wrapper on `DeclareLaunchArgument` to restrict the choices to the values of an enum"""

    def __init__(self, enum_type: Type[Enum], *args: Any, optional: bool = False, **kwargs: Any) -> None:
        choices = [str(e.value) for e in enum_type] + (
            [] if not optional else _OPTIONAL_CHOICES
        )  # typing: ignore[attr-defined]
        if "choices" in kwargs:
            raise KeyError("Cannot set `choices` for `DeclareEnumLaunchArgument`")
        if "default_value" in kwargs:
            default_value = kwargs["default_value"]
            if isinstance(default_value, enum_type):
                if default_value not in enum_type:
                    raise ValueError(
                        (
                            f"For an Enum Launch Argument of type {enum_type.__name__}, the `default_value` must be"
                            f" from {choices} or {list(enum_type)}"
                        ),
                    )
                kwargs["default_value"] = str(default_value.value)
            elif isinstance(default_value, str):
                if default_value not in choices:
                    raise ValueError(
                        (
                            f"For an Enum Launch Argument of type {enum_type.__name__}, the `default_value` must be"
                            f" from {choices} or {list(enum_type)}"
                        ),
                    )
            else:
                raise TypeError(
                    (
                        f"For an Enum Launch Argument of type {enum_type.__name__}, the `default_value` must be of type"
                        f" `str` or `{enum_type.__name__}`"
                    ),
                )

        super().__init__(*args, choices=choices, **kwargs)
