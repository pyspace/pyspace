import logging

# create logger for test output
logger = logging.getLogger('TestLogger')
logger.setLevel(logging.DEBUG)

loggingFileHandler = logging.FileHandler("unittest_log.txt")
loggingStreamHandler = logging.StreamHandler()
loggingStreamHandler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s")
loggingFileHandler.setFormatter(formatter)
loggingStreamHandler.setFormatter(formatter)

logger.addHandler(loggingFileHandler)
logger.addHandler(loggingStreamHandler)

