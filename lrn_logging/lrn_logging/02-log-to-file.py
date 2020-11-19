import logging

# logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)
# logging.basicConfig(filename='02-example.log', level=logging.DEBUG, filemode='a')  # append to file
logging.basicConfig(filename='02-example.log', level=logging.DEBUG, filemode='w')  # write to file (overwrite)
logging.debug('This message should go to the log file')
logging.info('So should this')
logging.warning('and this, too')
logging.error('and non-ASCII stuff, too, like Øresund and Malmö')
