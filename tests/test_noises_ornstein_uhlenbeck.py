import unittest
import math

import torch

from skrl.resources.noises.torch import OrnsteinUhlenbeckNoise


class TestCase(unittest.TestCase):
    def setUp(self):
        self.devices = ['cpu', 'cuda:0']

        self.sizes = [(1000, 2), [2000, 10, 1], torch.Size([3000])]
        self.thetas = (10 * (torch.rand(len(self.sizes)) + 0.5)).tolist()   # positive non-zero values
        self.sigmas = (10 * (torch.rand(len(self.sizes)) + 0.5)).tolist()   # positive non-zero values
        self.base_scales = (10 * (torch.rand(len(self.sizes)) + 0.5)).tolist()   # positive non-zero values

    def tearDown(self):
        pass

    def test_devices(self):
        for device in self.devices:
            noise = OrnsteinUhlenbeckNoise(theta=0.15, sigma=0.1, base_scale=1.0, device=device)
            self.assertEqual(noise.device, torch.device(device))

    def test_method_sample(self):
        for theta, sigma, base_scale in zip(self.thetas, self.sigmas, self.base_scales):
            # create noise
            noise = OrnsteinUhlenbeckNoise(theta=theta, sigma=sigma, base_scale=base_scale, device='cpu')
            # iterate over all sizes
            for size in self.sizes:
                # iterate 10 times
                for i in range(10):
                    # sample noise
                    output = noise.sample(size)
                    # check shape
                    self.assertEqual(output.size(), torch.Size(size))

    def test_method_sample_like(self):
        for theta, sigma, base_scale in zip(self.thetas, self.sigmas, self.base_scales):
            # create noise
            noise = OrnsteinUhlenbeckNoise(theta=theta, sigma=sigma, base_scale=base_scale, device='cpu')
            # iterate over all sizes
            for size in self.sizes:
                # create tensor
                tensor = torch.rand(size)
                # iterate 10 times
                for i in range(10):
                    # sample noise
                    output = noise.sample_like(tensor)
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
