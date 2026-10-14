import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agent.tools import calculator, write_file, read_file


def test_calculator_basic():
    assert calculator("2 + 2") == "4"
    assert calculator("10 * 4") == "40"
    assert calculator("2 ** 10") == "1024"


def test_calculator_rejects_garbage():
    result = calculator("__import__('os').system('echo hacked')")
    assert result.startswith("Error")


def test_calculator_division():
    assert calculator("9 / 2") == "4.5"


def test_file_roundtrip():
    write_file("test_output.txt", "hello world")
    assert read_file("test_output.txt") == "hello world"


def test_read_missing_file():
    result = read_file("does_not_exist.txt")
    assert result.startswith("Error")


def test_path_traversal_blocked():
    # Even if the model tries "../../etc/passwd", basename() strips the path
    write_file("../../evil.txt", "should be sandboxed")
    result = read_file("evil.txt")
    assert result == "should be sandboxed"


if __name__ == "__main__":
    test_calculator_basic()
    test_calculator_rejects_garbage()
    test_calculator_division()
    test_file_roundtrip()
    test_read_missing_file()
    test_path_traversal_blocked()
    print("All tests passed.")
