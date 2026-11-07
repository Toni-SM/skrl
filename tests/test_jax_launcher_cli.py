import sys
from unittest import TestCase, mock

from skrl.utils.distributed.jax import launcher


class JaxLauncherTests(TestCase):
    def test_invalid_arguments(self):
        for option, value in [("--nnodes", "0"), ("--nproc-per-node", "0"), ("--node-rank", "-1"), ("--coordinator-address", "host:70000")]:
            with self.subTest(option=option), mock.patch.object(sys, "argv", ["launcher", option, value, "train.py"]):
                with self.assertRaises(SystemExit) as error:
                    launcher.launch()
                self.assertEqual(error.exception.code, 2)

    def test_ipv6_endpoint(self):
        with mock.patch.object(sys, "argv", ["launcher", "--coordinator-address", "[::1]:1234", "train.py"]), mock.patch.object(launcher, "_start_processes") as spawn:
            launcher.launch()
        env = spawn.call_args.args[1][0]
        self.assertEqual((env["JAX_COORDINATOR_ADDR"], env["JAX_COORDINATOR_PORT"]), ("[::1]", "1234"))

    def test_worker_status(self):
        with self.assertRaises(SystemExit) as error:
            launcher._start_processes([sys.executable, "-c", "raise SystemExit(7)"], [{}], 1)
        self.assertEqual(error.exception.code, 7)
