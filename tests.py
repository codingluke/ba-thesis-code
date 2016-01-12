import unittest
import importlib

def test(coverage=False, testcase=None):
    """Run the unit tests."""
    suite = None
    if testcase:
      test_module = importlib.import_module("tests.%s" % testcase)
      suite = unittest.TestLoader().loadTestsFromModule(test_module)
    else:
        suite = unittest.TestLoader().discover('tests')
    unittest.TextTestRunner(verbosity=2).run(suite)

if __name__ == '__main__':
    test()
