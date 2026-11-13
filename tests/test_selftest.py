import unittest

from varamon.selftest import run


class SelfTest(unittest.TestCase):
    def test_generated_signals(self):
        self.assertEqual(run(verbose=False), 0)


if __name__ == "__main__":
    unittest.main()
