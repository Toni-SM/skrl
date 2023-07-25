import math
import unittest
import gym

import jax
import jax.numpy as jnp
import numpy as np

from skrl.memories.jax import Memory


class TestCase(unittest.TestCase):
    def setUp(self):
        self.devices = [jax.devices("cpu")[0], jax.devices("gpu")[0]]

        self.memory_sizes = [10, 100, 1000]
        self.num_envs = [1, 10, 100]

        self.names = ["states", "actions", "rewards", "dones"]
        self.raw_sizes = [gym.spaces.Box(-1, 1, shape=(5,)), gym.spaces.Discrete(5), 1, 1]
        self.sizes = [5, 1, 1, 1]
        self.raw_dtypes = [jnp.float32, int, float, bool]
        self.dtypes = [np.float32, np.int32, np.float32, bool]
        self.mini_batches = [1, 2, 3, 5, 7]

    def tearDown(self):
        pass

    def test_devices(self):
        for device in self.devices:
            # TODO: test
            pass

    def test_tensor_names(self):
        for memory_size, num_envs in zip(self.memory_sizes, self.num_envs):
            # create memory
            memory = Memory(memory_size=memory_size, num_envs=num_envs)

            # create tensors
            for name, size, dtype in zip(self.names, self.raw_sizes, self.raw_dtypes):
                memory.create_tensor(name, size, dtype)

            # test memory.get_tensor_names
            self.assertCountEqual(self.names, memory.get_tensor_names(), "get_tensor_names")

            # test memory.get_tensor_by_name
            for name, size, dtype in zip(self.names, self.sizes, self.dtypes):
                tensor = memory.get_tensor_by_name(name, keepdim=True)
                self.assertSequenceEqual(memory.get_tensor_by_name(name, keepdim=True).shape, (memory_size, num_envs, size), "get_tensor_by_name(..., keepdim=True)")
                self.assertSequenceEqual(memory.get_tensor_by_name(name, keepdim=False).shape, (memory_size * num_envs, size), "get_tensor_by_name(..., keepdim=False)")
                self.assertEqual(memory.get_tensor_by_name(name, keepdim=True).dtype, dtype, "get_tensor_by_name(...).dtype")

            # test memory.set_tensor_by_name
            for name, size, dtype in zip(self.names, self.sizes, self.raw_dtypes):
                new_tensor = jnp.arange(memory_size * num_envs * size).reshape(memory_size, num_envs, size).astype(dtype)
                memory.set_tensor_by_name(name, new_tensor)
                tensor = memory.get_tensor_by_name(name, keepdim=True)
                self.assertTrue((tensor == new_tensor).all().item(), "set_tensor_by_name(...)")

    def test_sample(self):
        for memory_size, num_envs in zip(self.memory_sizes, self.num_envs):
            # create memory
            memory = Memory(memory_size=memory_size, num_envs=num_envs)

            # create tensors
            for name, size, dtype in zip(self.names, self.raw_sizes, self.raw_dtypes):
                memory.create_tensor(name, size, dtype)

            # fill memory
            for name, size, dtype in zip(self.names, self.sizes, self.raw_dtypes):
                new_tensor = jnp.arange(memory_size * num_envs * size).reshape(memory_size, num_envs, size).astype(dtype)
                memory.set_tensor_by_name(name, new_tensor)

            # test memory.sample_all
            for i, mini_batches in enumerate(self.mini_batches):
                samples = memory.sample_all(self.names, mini_batches=mini_batches)
                for sample, name, size in zip(samples[i], self.names, self.sizes):
                    self.assertSequenceEqual(sample.shape, (memory_size * num_envs, size), f"sample_all(...).shape with mini_batches={mini_batches}")
                    tensor = memory.get_tensor_by_name(name, keepdim=True)
                    self.assertTrue((sample.reshape(memory_size, num_envs, size) == tensor).all().item(), f"sample_all(...) with mini_batches={mini_batches}")


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
