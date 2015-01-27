import doctest
import unittest

from ids_teach import models
from ids_teach import utils
from ids_teach import ids


class TestDocs(unittest.TestCase):
    def test_models_docs(self):
        doctest.DocTestSuite(models)

    def test_utils_docs(self):
        doctest.DocTestSuite(utils)

    def test_ids_docs(self):
        doctest.DocTestSuite(ids)


if __name__ == '__main__':
    unittest.main(verbosity=2)
