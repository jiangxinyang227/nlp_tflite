
import logging


def get_logger(filename):
    """
    
    """
 
    logging.basicConfig(level=logging.DEBUG,
                        filename=filename,
                        datefmt='%Y/%m/%d %H:%M:%S',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')
    logger = logging.getLogger("log")

    return logger
    