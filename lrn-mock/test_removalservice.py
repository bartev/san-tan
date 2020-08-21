#!/usr/bin/env python


from removalservice import RemovalService

import unittest.mock as mock
import unittest


class RemovalServiceTestCase(unittest.TestCase):
    """Documentation for RmTestCase

    """
    print('in test_remvovalservice.py')

    @mock.patch('mymodule.os.path')
    @mock.patch('mymodule.os')
    def test_rm(self, mock_os, mock_path):

        # instantiate our service
        reference = RemovalService()

        # test that the remove call was NOT called
        # set up the mock
        mock_path.isfile.return_value = False
        reference.rm('any path')
        self.assertFalse(mock_os.remove.called, 'Failed to not remove the file if not present')

        # # test that rm called os.remove wiht the right parameters
        # # make the file 'exist'
        mock_path.isfile.return_value = True
        print(f'mock_path.isfile.return_value: {mock_path.isfile.return_value}')
        reference.rm('any path')
        mock_os.remove.assert_called_with('any path')
