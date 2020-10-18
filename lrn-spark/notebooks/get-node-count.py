#!/usr/bin/env python

from os import environ

import pprint
import platform
import pyspark

pp = pprint.PrettyPrinter(indent=4)

spark = pyspark.sql.SparkSession.builder.appName('test').getOrCreate()

spark._jsc.sc().setLogLevel('WARN')

print(f'platform.python_version(): {platform.python_version()}')
print(f'platform.mac_ver():        {platform.mac_ver()}')


def testfunc(num: int) -> str:
    return "type annotations look ok"


print(f'testfunc(1) = {testfunc(1)}')

num_nodes = spark._jsc.sc().getExecutorMemoryStatus().keySet().size()
print(f'\n\nYou are using {num_nodes} nodes in this session\n\n')

print('spark.sparkContext._conf.getAll()')
pp.pprint(spark.sparkContext._conf.getAll())

home_env = {k: v for k, v in environ.items() if 'HOME' in k}
print('HOME env variables')
pp.pprint(home_env)
