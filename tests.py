import unittest
import importlib
import pdb
# import tests.test_metric

def test(coverage=False, testcase=None):
    """Run the unit tests."""
    suite = None
    if testcase:
      print "tests.%s" % testcase
      test_module = importlib.import_module("tests.%s" % testcase)
      suite = unittest.TestLoader().loadTestsFromModule(test_module)
    else:
        suite = unittest.TestLoader().discover('tests')
    unittest.TextTestRunner(verbosity=2).run(suite)

if __name__ == '__main__':
    # pdb.set_trace()
    test()
