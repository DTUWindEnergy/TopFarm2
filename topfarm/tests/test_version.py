import re


def test_version_exists():
    from topfarm import __version__

    assert __version__ is not None


def test_version_format():
    from topfarm import __version__

    # Regex pattern to match semantic versioning (e.g., 1.0.0, 2.1.3 or 3.0.0.postX)
    pattern = r"^\d+\.\d+\.\d+(\.post\d+)?$"
    assert re.match(
        pattern, __version__
    ), f"Version {__version__} does not match the pattern {pattern}"
