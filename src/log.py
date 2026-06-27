# importing libraries like logging, os, datetime
import logging
import os
from datetime import datetime
# *************************
LOG_FILE =f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# give path 
path = os.path.join(os.getcwd(),'logs',LOG_FILE)

os.makedirs(path,exist_ok=True)

file_path = os.path.join(path,LOG_FILE)

#logging setup
logging.basicConfig(
    filename=file_path,
    format="[%(asctime)s] %(lineno)d %(name)s -%(levelname)s -%(message)s",
    level=logging.INFO
)

# to checking CustomeException
if __name__ == '__main__':
    logging.info("logging is working")

