"""Tests for skrl.utils.postprocessing module.

Covers MemoryFileIterator (numpy, torch, csv formats) and TensorboardFileIterator.

Notable bug exposed:
    In MemoryFileIterator._format_csv, boolean values are parsed with ``bool(item)``
    which evaluates ``bool("False") == True`` because any non-empty string is truthy.
    The correct expression is ``item == "True"``.

    The CSV format exported by skrl's Memory.save() writes tensor names in sorted
    (alphabetical) order, so the parser's assumption of sorted column groups is valid.
    However, the boolean conversion is still broken for "False" values.
"""
from __future__ import annotations

import csv
import os
import tempfile

import numpy as np
import pytest
import torch

from skrl.utils.postprocessing import MemoryFileIterator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_npz(path: str, data: dict) -> None:
    """Write a .npz file (path should NOT include the .npz extension)."""
    np.savez(path, **data)


def _write_pt(path: str, data: dict) -> None:
    """Write a .pt file."""
    torch.save(data, path)


def _write_csv(path: str, header: list[str], rows: list[list]) -> None:
    """Write a CSV file with the given header and rows.

    Note: skrl's Memory.save() writes tensor names in sorted (alphabetical) order.
    Tests that exercise the CSV parser must respect this convention so that the
    column-index mapping computed from the header matches the actual column layout.
    """
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# MemoryFileIterator – empty glob
# ---------------------------------------------------------------------------


def test_memory_file_iterator_empty_glob():
    """Iterator over a non-matching glob should raise StopIteration immediately."""
    it = MemoryFileIterator("/tmp/__no_such_files_skrl_test_*.npz")
    with pytest.raises(StopIteration):
        next(it)


def test_memory_file_iterator_is_iterable():
    """Iterator should return itself from __iter__."""
    it = MemoryFileIterator("/tmp/__no_such_files_skrl_test_*.npz")
    assert iter(it) is it


# ---------------------------------------------------------------------------
# MemoryFileIterator – NumPy format
# ---------------------------------------------------------------------------


def test_memory_file_iterator_numpy_single_file():
    """Single .npz file should be loaded and returned with correct filename and data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        npz_path = os.path.join(tmpdir, "memory_0")
        _write_npz(npz_path, {"states": arr})

        results = list(MemoryFileIterator(os.path.join(tmpdir, "*.npz")))
        assert len(results) == 1
        filename, data = results[0]
        assert filename == "memory_0.npz"
        np.testing.assert_array_equal(data["states"], arr)


def test_memory_file_iterator_numpy_multiple_files_sorted():
    """Multiple .npz files should be yielded in sorted (alphabetical) order."""
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(3):
            _write_npz(
                os.path.join(tmpdir, f"mem_{i}"),
                {"x": np.array([i], dtype=np.float32)},
            )

        results = list(MemoryFileIterator(os.path.join(tmpdir, "*.npz")))
        assert len(results) == 3
        filenames = [r[0] for r in results]
        assert filenames == sorted(filenames), "Files must be yielded in sorted order"
        for i, (_, data) in enumerate(results):
            np.testing.assert_array_equal(data["x"], np.array([i], dtype=np.float32))


def test_memory_file_iterator_numpy_multiple_tensors():
    """A .npz file with multiple tensors should return all of them."""
    with tempfile.TemporaryDirectory() as tmpdir:
        states = np.ones((4, 2, 3), dtype=np.float32)
        actions = np.zeros((4, 2, 1), dtype=np.float32)
        _write_npz(os.path.join(tmpdir, "mem"), {"states": states, "actions": actions})

        results = list(MemoryFileIterator(os.path.join(tmpdir, "*.npz")))
        assert len(results) == 1
        _, data = results[0]
        assert set(data.keys()) == {"states", "actions"}
        np.testing.assert_array_equal(data["states"], states)
        np.testing.assert_array_equal(data["actions"], actions)


# ---------------------------------------------------------------------------
# MemoryFileIterator – PyTorch format
# ---------------------------------------------------------------------------


def test_memory_file_iterator_torch_single_file():
    """Single .pt file should be loaded and returned with correct filename and data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tensor = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
        pt_path = os.path.join(tmpdir, "memory_0.pt")
        _write_pt(pt_path, {"states": tensor})

        results = list(MemoryFileIterator(os.path.join(tmpdir, "*.pt")))
        assert len(results) == 1
        filename, data = results[0]
        assert filename == "memory_0.pt"
        assert torch.equal(data["states"], tensor)


def test_memory_file_iterator_torch_multiple_files_sorted():
    """Multiple .pt files should be yielded in sorted order."""
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(3):
            _write_pt(
                os.path.join(tmpdir, f"mem_{i}.pt"),
                {"val": torch.tensor([float(i)])},
            )

        results = list(MemoryFileIterator(os.path.join(tmpdir, "*.pt")))
        assert len(results) == 3
        filenames = [r[0] for r in results]
        assert filenames == sorted(filenames), "Files must be yielded in sorted order"
        for i, (_, data) in enumerate(results):
            assert torch.equal(data["val"], torch.tensor([float(i)]))


# ---------------------------------------------------------------------------
# MemoryFileIterator – CSV format
# ---------------------------------------------------------------------------


def test_memory_file_iterator_csv_float_values():
    """CSV with float columns should be parsed correctly.

    Column groups must be written in sorted (alphabetical) order to match the
    index mapping computed by the parser (which also sorts names).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "memory_0.csv")
        # "actions" sorts before "states", so actions columns come first
        header = ["actions.0", "states.0", "states.1"]
        rows = [["3.0", "1.0", "2.0"], ["6.0", "4.0", "5.0"]]
        _write_csv(csv_path, header, rows)

        results = list(MemoryFileIterator(os.path.join(tmpdir, "*.csv")))
        assert len(results) == 1
        filename, data = results[0]
        assert filename == "memory_0.csv"
        assert "states" in data
        assert "actions" in data
        assert data["actions"] == [[3.0], [6.0]]
        assert data["states"] == [[1.0, 2.0], [4.0, 5.0]]


def test_memory_file_iterator_csv_bool_true_value():
    """CSV boolean 'True' should be parsed as True (bool)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "memory_bool.csv")
        header = ["terminated.0"]
        rows = [["True"], ["True"]]
        _write_csv(csv_path, header, rows)

        results = list(MemoryFileIterator(os.path.join(tmpdir, "*.csv")))
        _, data = results[0]
        assert data["terminated"] == [[True], [True]]


def test_memory_file_iterator_csv_bool_false_value():
    """CSV boolean 'False' should be parsed as False (bool).

    This test exposes a real bug in the original implementation:

        ``bool("False") == True``

    because any non-empty string is truthy in Python.
    The correct fix is to use ``item == "True"`` instead of ``bool(item)``.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "memory_bool_false.csv")
        header = ["terminated.0"]
        rows = [["False"], ["False"]]
        _write_csv(csv_path, header, rows)

        results = list(MemoryFileIterator(os.path.join(tmpdir, "*.csv")))
        _, data = results[0]
        # Each row should contain False, not True
        assert data["terminated"] == [[False], [False]], (
            "bool('False') incorrectly evaluates to True in Python. "
            "The CSV parser must use `item == 'True'` instead of `bool(item)`."
        )


def test_memory_file_iterator_csv_mixed_bool_values():
    """CSV with mixed True/False booleans should parse each correctly.

    This test also exposes the ``bool("False") == True`` bug: rows containing
    "False" will be incorrectly parsed as True with the current implementation.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "memory_mixed.csv")
        header = ["done.0"]
        rows = [["True"], ["False"], ["True"], ["False"]]
        _write_csv(csv_path, header, rows)

        results = list(MemoryFileIterator(os.path.join(tmpdir, "*.csv")))
        _, data = results[0]
        assert data["done"] == [[True], [False], [True], [False]], (
            "Mixed True/False values must be parsed correctly. "
            "bool('False') == True is a known Python gotcha; use `item == 'True'`."
        )


def test_memory_file_iterator_csv_invalid_header_returns_empty():
    """CSV with a malformed header (no dot separator) should return empty dict."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "bad_header.csv")
        # Header without the required "name.index" format
        _write_csv(csv_path, ["states", "actions"], [["1.0", "2.0"]])

        results = list(MemoryFileIterator(os.path.join(tmpdir, "*.csv")))
        assert len(results) == 1
        filename, data = results[0]
        assert filename == "bad_header.csv"
        assert data == {}


def test_memory_file_iterator_csv_empty_data_rows():
    """CSV with a valid header but no data rows should return empty lists per key."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "empty.csv")
        _write_csv(csv_path, ["actions.0", "states.0"], [])

        results = list(MemoryFileIterator(os.path.join(tmpdir, "*.csv")))
        assert len(results) == 1
        _, data = results[0]
        assert data["states"] == []
        assert data["actions"] == []


# ---------------------------------------------------------------------------
# MemoryFileIterator – unsupported format
# ---------------------------------------------------------------------------


def test_memory_file_iterator_unsupported_format_raises():
    """Unsupported file extension should raise ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        txt_path = os.path.join(tmpdir, "memory.txt")
        with open(txt_path, "w") as f:
            f.write("dummy")

        it = MemoryFileIterator(os.path.join(tmpdir, "*.txt"))
        with pytest.raises(ValueError, match="Unsupported format"):
            next(it)


# ---------------------------------------------------------------------------
# MemoryFileIterator – iterator protocol
# ---------------------------------------------------------------------------


def test_memory_file_iterator_exhaustion():
    """After all files are consumed, subsequent next() calls raise StopIteration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_npz(os.path.join(tmpdir, "mem"), {"x": np.array([1.0])})

        it = MemoryFileIterator(os.path.join(tmpdir, "*.npz"))
        next(it)  # consume the single file
        with pytest.raises(StopIteration):
            next(it)


def test_memory_file_iterator_for_loop():
    """MemoryFileIterator should work correctly in a for loop."""
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(3):
            _write_npz(os.path.join(tmpdir, f"mem_{i}"), {"v": np.array([float(i)])})

        count = 0
        for filename, data in MemoryFileIterator(os.path.join(tmpdir, "*.npz")):
            assert filename.endswith(".npz")
            assert "v" in data
            count += 1
        assert count == 3


def test_memory_file_iterator_mixed_formats_separate_globs():
    """Separate globs for .npz and .pt files should each return the correct files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        arr = np.ones((2, 3, 1), dtype=np.float32)
        _write_npz(os.path.join(tmpdir, "a_mem"), {"x": arr})
        _write_pt(os.path.join(tmpdir, "b_mem.pt"), {"y": torch.ones(2, 3, 1)})

        npz_results = list(MemoryFileIterator(os.path.join(tmpdir, "*.npz")))
        pt_results = list(MemoryFileIterator(os.path.join(tmpdir, "*.pt")))

        assert len(npz_results) == 1
        assert len(pt_results) == 1
        assert npz_results[0][0].endswith(".npz")
        assert pt_results[0][0].endswith(".pt")
