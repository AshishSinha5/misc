# https://docs.python.org/3/howto/logging.html
import logging 

#logging.warning("Watch Out!! New logger learner") # will print in cosole
#logging.info("Information -- everything is chill") # will not print anything 


def test():
    logging.basicConfig(filename = "using-a-logger.log", level=logging.INFO)
    logging.info("started")
    for i in range(10):
        logging.info(f"{i}")
    logging.warning("hell yeah")
    logging.info("ended")


if __name__ == '__main__':
    #logging.info('in main')
    test()
    #logging.info('yo ended main')


