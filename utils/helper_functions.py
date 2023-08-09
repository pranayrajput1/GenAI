import logging
import time
import psutil

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


def get_log():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    return logging


def get_time(start_time, end_time):
    elapsed_time_minutes = (end_time - start_time) / 60

    logging.info(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    logging.info(f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    logging.info(f"Elapsed Time: {elapsed_time_minutes:.2f}")


def get_memory_usage():
    memory = psutil.virtual_memory()
    total_memory = memory.total / (1024 ** 3)  # Convert bytes to gigabytes
    used_memory = memory.used / (1024 ** 3)
    available_memory = memory.available / (1024 ** 3)
    percentage_used = memory.percent

    logging.info(f"Total Memory: {total_memory:.2f} GB")
    logging.info(f"Used Memory: {used_memory:.2f} GB")
    logging.info(f"Available Memory: {available_memory:.2f} GB")
    logging.info(f"Percentage Used: {percentage_used}%")
