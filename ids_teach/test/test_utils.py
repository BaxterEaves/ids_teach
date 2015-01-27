import unittest
import numpy as np
from ids_teach import utils


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
