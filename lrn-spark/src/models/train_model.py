def make_string_indexer(col_name):
    """valid values of handleInvalid
    skip (filter rows)
    error (throw an error)
    keep (put in a special additional bucket)

    NOTE: spark 3.0 will accept multple columns as input/output
    """
    from pyspark.ml.feature import StringIndexer

    encoded_col_name = f'{col_name}_Index'
    string_indexer = StringIndexer(
        inputCol=col_name, outputCol=encoded_col_name, handleInvalid='keep'
    )
    return string_indexer

def make_string_indexer_list(col_names):
    """make a list of StringIndexers"""
    return [make_string_indexer(c) for c in col_names]

def make_one_hot_encoder(col_names):
    """each `*_OHE` column will be a SparseVector after fitting and transformation

    Usage:
    ohe_room_type = make_one_hot_encoder(['room_type'])
    encoded_room_type = ohe_room_type.fit(transformed_room_type)

    encoded_room_type.transform(transformed_room_type).show()

    +---------------+-----+---------------+-------------+
    |      room_type|price|room_type_Index|room_type_OHE|
    +---------------+-----+---------------+-------------+
    |   Private room|200.0|            1.0|(3,[1],[1.0])|
    |Entire home/apt|250.0|            0.0|(3,[0],[1.0])|
    |Entire home/apt|250.0|            0.0|(3,[0],[1.0])|
    """
    input_col_names = [f'{col_name}_Index' for col_name in col_names]
    output_col_names = [f'{col_name}_OHE' for col_name in col_names]
    estimator = OneHotEncoderEstimator(
        inputCols=input_col_names, outputCols=output_col_names
    )
    return estimator
