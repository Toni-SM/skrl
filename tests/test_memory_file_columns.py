import tempfile
import unittest
from pathlib import Path

import numpy as np

from skrl.utils.postprocessing import MemoryFileIterator


class MemoryFileColumnTest(unittest.TestCase):
    def csv(self, header, values):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.csv"
            path.write_text(header + "\n" + values + "\n")
            return next(MemoryFileIterator(str(path)))[1]

    def test_false(self):
        self.assertEqual(self.csv("terminated.0", "False")["terminated"], [[False]])

    def test_reordered_names(self):
        self.assertEqual(self.csv("z.0,a.0", "3,9")["z"], [[3]])

    def test_interleaved_components(self):
        self.assertEqual(self.csv("a.1,b.0,a.0", "3,9,6")["a"], [[6, 3]])

    def test_dotted_names(self):
        self.assertEqual(self.csv("agent.reward.0", "7")["agent.reward"], [[7]])

    def test_archive_returns_dictionary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.npz"
            np.savez(path, value=[42])
            data = next(MemoryFileIterator(str(path)))[1]
            self.assertIsInstance(data, dict)
            self.assertEqual(data["value"].item(), 42)
