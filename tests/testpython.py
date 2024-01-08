"""Test for python fonctions."""

import unittest
import UnitTest.classif_animals10

class TestConcat(unittest.TestCase):
    """Test for concat fonction."""

    def testconcat(self):
        """Test with 5236."""
        r = UnitTest.classif_animals10.concat([5,2,3,6])
        self.assertEqual(r, "cinq mille deux cent trente six")

if __name__ == "__main__":
    unittest.main()
