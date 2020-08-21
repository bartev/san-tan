#!/usr/bin/env python


from mymodule import rm

import os.path
import tempfile
import unittest


class RmTestCase(unittest.TestCase):
    """Documentation for RmTestCase

    """
    print('in test_mymodule.py')

    tmpfilepath = os.path.join(tempfile.gettempdir(), 'tmp-testfile')

    def setUp(self):
        print(f'setUp: tmpfilepath: {self.tmpfilepath}')
        with open(self.tmpfilepath, 'w') as f:
            f.write('Delete me!')

    def test_rm(self):
        print(f'test_rm: tmpfilepath: {self.tmpfilepath}')

        # test that it's there
        self.assertTrue(os.path.isfile(self.tmpfilepath), 'File does not exist')

        # remove the filename
        rm(self.tmpfilepath)

        # test that it was actually removed
        self.assertFalse(os.path.isfile(self.tmpfilepath), 'Failed to remove the file')
