# imports for adhoc notebooks
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:95% !important; }</style>"))

# magics
%load_ext blackcellmagic
# start cell with `%% black` to format using `black`

%load_ext autoreload
# start cell with `%autoreload` to reload module
# https://ipython.org/ipython-doc/stable/config/extensions/autoreload.html

# imports

import logging
import numpy as np
import os
import os.path as path
import pandas as pd
import sklearn.metrics as skm

import ai.applecare.utilities.common as uc
import ai.applecare.utilities.constants as cst
import ai.applecare.utilities.data_pull as dp
import ai.applecare.utilities.file_io as io
import ai.applecare.utilities.formatting as fm
import ai.applecare.utilities.log as lg

from importlib import reload

lg.setup_logging()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# import ai.teradata
# import ai.teradata.tools as tt
# from ai.teradata import td

edw = dp.get_connection()
