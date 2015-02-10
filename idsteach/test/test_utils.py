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


import unittest
import numpy as np
from idsteach import utils


class TestBellNUmber(unittest.TestCase):
    def test_bell_0(self):
        self.assertEqual(utils.bell_number(0), 1)

    def test_bell_1(self):
        self.assertEqual(utils.bell_number(1), 1)

    def test_bell_4(self):
        self.assertEqual(utils.bell_number(4), 15)

    def test_bell_10(self):
        self.assertEqual(utils.bell_number(10), 115975)

    def test_bell_12(self):
        self.assertEqual(utils.bell_number(12), 4213597)

    def test_bell_14(self):
        self.assertEqual(utils.bell_number(14), 190899322)


class TestParitionGenerator(unittest.TestCase):
    def setUp(self):
        self.genparts = lambda n: [p for p in utils.partition_generator(n)]

    def should_generate_correct_number_of_partitions_n4(self):
        parts = self.genparts(4)
        self.assertEqual(len(parts), 15)

    def should_generate_correct_number_of_partitions_n2(self):
        parts = self.genparts(2)
        self.assertEqual(len(parts), 3)

    def should_generate_correct_number_of_partitions_n8(self):
        parts = self.genparts(8)
        self.assertEqual(len(parts), 4140)

    def all_partitions_should_be_unique_n4(self):
        parts = self.genparts(4)
        tuple_parts = [tuple(p) for p in parts]
        set.assertEqual(len(set(tuple_parts)), 15)

    def all_partitions_should_be_unique_n7(self):
        parts = self.genparts(7)
        tuple_parts = [tuple(p) for p in parts]
        set.assertEqual(len(set(tuple_parts)), 877)


if __name__ == '__main__':
    unittest.main()
