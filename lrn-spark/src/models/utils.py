def parse_param_map(pm):
    """parse the ParamMap and return a dict of name: val

    Usage:

    > parse_param_map(model.extractParamMap())

    {'cacheNodeIds': False,
     'checkpointInterval': 10,
     'featureSubsetStrategy': 'auto',
     'featuresCol': 'features',
     'impurity': 'variance',
     'labelCol': 'price',
     'maxBins': 40,
     'maxDepth': 6,
     'maxMemoryInMB': 256,
     'minInfoGain': 0.0,
     'minInstancesPerNode': 1,
     'numTrees': 100,
     'predictionCol': 'prediction',
     'seed': 42,
     'subsamplingRate': 1.0}
    """
    return {m.name: pm[m] for m in pm}
