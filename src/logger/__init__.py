import logging
from datetime import datetime
import os
import shutil

LOG_DIR = "logs"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

def get_log_file_name():
    return f"log_{TIMESTAMP}.log"


LOG_FILE_NAME = get_log_file_name()

if os.path.exists(LOG_DIR):
    shutil.rmtree(LOG_DIR)
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE_NAME)

logging.basicConfig(
                    format='[%(asctime)s] \t%(levelname)s \t%(lineno)d \t%(filename)s \t%(funcName)s() \t%(message)s',
                    level=logging.INFO,
                    force=True,
                    handlers=[
                        logging.FileHandler(LOG_FILE_PATH),
                        #logging.StreamHandler()
                    ]
                    )

logger = logging.getLogger("QP-Chatbot")
