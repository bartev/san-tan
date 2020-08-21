#!/usr/bin/env python


from mymodule import rm

import unittest.mock as mock
import unittest


class RmTestCase(unittest.TestCase):
    """Documentation for RmTestCase

    """
    print('in test_mymodule_mock.py')

    @mock.patch('mymodule.os.path')
    @mock.patch('mymodule.os')
    def test_rm(self, mock_os, mock_path):
        # set up the mock
        mock_path.isfile.return_value = False

        rm('any path')

        # test that the remove call was NOT called
        self.assertFalse(mock_os.remove.called, 'Failed to not remove the file if not present')

        # make the file 'exist'
        mock_path.isfile.return_value = True

        # test that rm called os.remove wiht the right parameters
        rm('any path')
        mock_os.remove.assert_called_with('any path')
