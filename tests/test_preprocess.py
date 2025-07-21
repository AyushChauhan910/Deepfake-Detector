import unittest
import numpy as np
from deepfake_detector.preprocess import validate_input_file
from pathlib import Path

class TestPreprocess(unittest.TestCase):
    
    def test_validate_input_file(self):
        # This would need actual test files
        # For now, just test the ValueError case
        with self.assertRaises(ValueError):
            validate_input_file("test.xyz", ['.jpg', '.png'])

if __name__ == '__main__':
    unittest.main()
