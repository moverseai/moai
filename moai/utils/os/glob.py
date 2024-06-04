import glob
import typing

__all__ = ["os_glob"]

os_glob = None


def sorted_glob(pattern: str, recursive=False) -> typing.Sequence[str]:
    return sorted(glob.glob(pattern, recursive=recursive))


if os_glob is None:
    from sys import platform

    if platform == "linux" or platform == "linux2":
        os_glob = sorted_glob
    elif platform == "darwin":
        os_glob = sorted_glob
    elif platform == "win32":
        os_glob = glob.glob
