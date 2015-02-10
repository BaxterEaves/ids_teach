# IDSTeach: Generate data to teach continuous categorical data.
# Copyright (C) 2015  Baxter S. Eaves Jr.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import doctest
import unittest

from idsteach import models
from idsteach import utils
from idsteach import ids


class TestDocs(unittest.TestCase):
    def test_models_docs(self):
        doctest.DocTestSuite(models)

    def test_utils_docs(self):
        doctest.DocTestSuite(utils)

    def test_ids_docs(self):
        doctest.DocTestSuite(ids)


if __name__ == '__main__':
    unittest.main(verbosity=2)
