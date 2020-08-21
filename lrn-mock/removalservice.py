#!/usr/bin/env python

import os
import os.path


class RemovalService(object):
    """Documentation for RemovalService

    """

    def rm(self, filename):
        if os.path.isfile(filename):
            os.remove(filename)
