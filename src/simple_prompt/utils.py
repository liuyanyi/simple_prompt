import datetime


def is_debug() -> bool:
    """
    Check if the code is running in debug mode.

    Examples:
    >>> is_debug()
    False
    """
    import sys

    gettrace = getattr(sys, "gettrace", None)
    if gettrace is None:
        return False
    elif gettrace():
        return True
    else:
        return False


def beautify_time(timestamp: float, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Convert a timestamp to a human-readable string.

    Examples:
    >>> beautify_time(1629999999.999)
    '2021-08-26 08:53:19'
    """
    if timestamp is None:
        return ""
    return datetime.datetime.fromtimestamp(timestamp).strftime(fmt)
