from typing import Tuple

import os
import csv
import glob
import numpy as np

import torch
import collections


class MemoryFileIterator():
    def __init__(self, directory: str, format: str) -> None:
        """Python iterator for loading data from exported memories
        
        The iterator will load the next memory file in the directory.
        The output of the iterator is a tuple of the filename and the memory data 
        where the memory data is a dictionary of torch.Tensor (PyTorch), numpy.ndarray (NumPy)
        or lists (CSV) depending on the format and the keys of the dictionary are the names of the variables

        Supported formats:

        - PyTorch (pt)
        - NumPy (npz)
        - Comma-separated values (csv)

        Expected data shapes:

        - PyTorch: (batch_size, num_channels, height, width)

        :param directory: Directory containing the exported memories
        :type directory: str
        :param format: Format of the exported memory (csv, npz, pt)
        :type format: str
        """
        self.n = 0
        self.format = format

        self.file_paths = glob.glob(directory + '/*.' + format)

    def __iter__(self) -> 'MemoryFileIterator':
        """Return self to make iterable"""
        return self

    def __next__(self) -> Tuple[str, dict]:
        """Return next batch

        :return: Tuple of filename and data
        :rtype: tuple
        """
        if self.n >= len(self.file_paths):
            raise StopIteration
        
        if self.format == 'pt':
            return self._format_torch()
        elif self.format == 'npz':
            return self._format_numpy()
        elif self.format == 'csv':
            return self._format_csv()
        else:
            raise ValueError("Unsupported format: {}. Available formats: pt, csv, npz".format(format))

    def _format_numpy(self) -> Tuple[str, dict]:
        """Load numpy array from file
        
        :return: Tuple of filename and data
        :rtype: tuple
        """
        filename = os.path.basename(self.file_paths[self.n])
        data = np.load(self.file_paths[self.n])

        self.n += 1
        return filename, data

    def _format_torch(self) -> Tuple[str, dict]:
        """Load PyTorch tensor from file

        :return: Tuple of filename and data
        :rtype: tuple
        """
        filename = os.path.basename(self.file_paths[self.n])
        data = torch.load(self.file_paths[self.n])

        self.n += 1
        return filename, data

    def _format_csv(self) -> Tuple[str, dict]:
        """Load CSV file from file

        :return: Tuple of filename and data
        :rtype: tuple
        """
        filename = os.path.basename(self.file_paths[self.n])

        with open(self.file_paths[self.n], 'r') as f:
            reader = csv.reader(f)
            
            # parse header
            try:
                header = next(reader, None)
                data = collections.defaultdict(int)
                for h in header:
                    h.split(".")[1]  # check header format
                    data[h.split(".")[0]] += 1
                names = sorted(list(data.keys()))
                sizes = [data[name] for name in names]
                indexes = [(low, high) for low, high in zip(np.cumsum(sizes) - np.array(sizes), np.cumsum(sizes))]
            except:
                self.n += 1
                return filename, {}

            # parse data
            data = {name: [] for name in names}
            for row in reader:
                for name, index in zip(names, indexes):
                    data[name].append([float(item) if item not in ["True", "False"] else bool(item) \
                        for item in row[index[0]:index[1]]])

        self.n += 1
        return filename, data
