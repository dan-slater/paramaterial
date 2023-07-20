"""For testing the download example function."""

import os
import shutil
import unittest


class TestExample(unittest.TestCase):
    """Test the download_example function."""

    def test_download_example(self):
        """Test the download_example function."""
        from paramaterial import example
        example.download_example(to_directory='test_examples', example_name='example_1')
        self.assertTrue(os.path.exists('test_examples/example_1'))
        # shutil.rmtree('test_examples')

