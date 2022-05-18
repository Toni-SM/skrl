import unittest
import math

import torch

from skrl.resources.noises.torch import GaussianNoise


class TestCase(unittest.TestCase):
    def setUp(self):
        self.devices = ['cpu', 'cuda:0']

        self.sizes = [(1000, 2), [2000, 10, 1], torch.Size([3000])]
        self.means = (10 * (torch.rand(len(self.sizes)) + 0.5) * torch.sign(torch.rand(len(self.sizes)) - 0.5)).tolist()
        self.stds = (10 * (torch.rand(len(self.sizes)) + 0.1)).tolist()   # positive non-zero values

    def tearDown(self):
        pass

    def test_devices(self):
        for device in self.devices:
            noise = GaussianNoise(mean=0, std=1.0, device=device)
            self.assertEqual(noise.device, torch.device(device))

    def test_method_sample(self):
        for mean, std in zip(self.means, self.stds):
            # create noise
            noise = GaussianNoise(mean=mean, std=std, device='cpu')
            # iterate over all sizes
            for size in self.sizes:
                # iterate 10 times
                for i in range(10):
                    # sample noise
                    output = noise.sample(size)
                    # check output
                    _mean = output.mean().item()
                    _std = output.std().item()
                    self.assertTrue(math.isclose(_mean, mean, rel_tol=abs(mean) * 0.25))
                    self.assertTrue(math.isclose(_std, std, rel_tol=std * 0.25))
                    # check shape
                    self.assertEqual(output.size(), torch.Size(size))

    def test_method_sample_like(self):
        for mean, std in zip(self.means, self.stds):
            # create noise
            noise = GaussianNoise(mean=mean, std=std, device='cpu')
            # iterate over all sizes
            for size in self.sizes:
                # create tensor
                tensor = torch.rand(size)
                # iterate 10 times
                for i in range(10):
                    # sample noise
                    output = noise.sample_like(tensor)
                    # check output
                    _mean = output.mean().item()
                    _std = output.std().item()
                    self.assertTrue(math.isclose(_mean, mean, rel_tol=abs(mean) * 0.25))
                    self.assertTrue(math.isclose(_std, std, rel_tol=std * 0.25))
                    # check shape
                    self.assertEqual(output.size(), torch.Size(size))


if __name__ == '__main__':
    import sys

    if not sys.argv[-1] == '--debug':
        raise RuntimeError('Test can only be runned manually with --debug flag')
    
    test = TestCase()
    test.setUp()
    for method in dir(test):
        if method.startswith('test_'):
            print('Running test: {}'.format(method))
            getattr(test, method)()
    test.tearDown()

    print('All tests passed.')
