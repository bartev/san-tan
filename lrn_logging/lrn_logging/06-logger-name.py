import logging

logger = logging.getLogger(__name__)

logger.warning('from logger')
logging.warning('logging')
print(type(logger))
