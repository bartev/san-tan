"""Utility functions for using spark"""


def get_mleap_jars(path_to_jars='/Users/bartev/dev/jars/mleap-lib/'):
    """Return a string with the jars needed to use ``mleap`` with spark
    the full path of all the jars in the directory will be combined in a
    string to use when getting a ``SparkSession``

    :param path_to_jars: the directory containing the mleap jars
    :returns: string containig paths of all jars
    :rtype: string

    Usage:
    mleap_jars = get_mleap_jars():
    spark = (
        SparkSession
        .builder
        .appName('foobar')
        .config('spark.jars', mleap_jars)
        .getOrCreate()
        )
    """
    import os
    import os.path as path

    p = path.abspath(path.expanduser(path_to_jars))
    jars = [path.join(p, j) for j in os.listdir(p) if j.endswith('jar')]
    return ','.join(jars)


def create_mleap_fname(fname, jar_path):
    """start file with 'jar:file', add dir path, end in '.zip' """
    import os.path as path
    import os

    full_fname = os.path.abspath(os.path.expanduser(os.path.join(jar_path, fname)))
    return f'jar:file:{full_fname}'
