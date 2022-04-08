# coding=utf-8
import sys
import os
import resource
from loguru import logger

# the code moved to function so it isn't automatically called.
# side effect: wipes loguru config and sets it as described below
def configure_default_logging():
    logger_short_format = "<cyan>MM</cyan> {time:HH:mm:ss} {extra[mem]} <level>{message}</level>"
    logger_long_format  = "<cyan>MM</cyan> {time:YYYY:MM:DD HH:mm:ss} | {module}:{file}:{function}:{line} | {level} | {extra[mem]} {message}"

    loguru_config = {
        "handlers": [
            {"sink": sys.stdout, "format": logger_short_format},
            {"sink": "logs/{time:YYYY/MM/DD/HH_CCC}.log", "enqueue": True, "delay": True,
             "format": logger_long_format},
        ],
        "levels": [
            {"name": "DEBUG", "color": "<fg #377eb8>"},
            {"name": "INFO", "color": "<fg #4daf4a>"},
            {"name": "WARNING", "color": "<fg #ff7f00>"},
            {"name": "ERROR", "color": "<fg #de2d26>"},
            {"name": "CRITICAL", "color": "<fg #a50f15>"},
            {"name": "SUCCESS", "color": "<cyan>"}
        ],
        "extra": {"mem": "@@@"}    # needed for memory logging
    }

    logger.configure(**loguru_config)

# activate the MM default logging config if env var is set.
# e.g. `$ export LOGURU_NodeCoder=1`
if os.environ.get("LOGURU_NodeCoder"):
    configure_default_logging()


# The below will cause memory use to be logged, but only if a newer version of loguru is installed:
#   pip install loguru -U
# If not installed, it will gracefully regress to the uninformative but decorative '@@@' from before.

if hasattr(logger, "patch"):   # patch does not exist in the ancient version (2.5 vs 4.1) we still use
    logger = logger.patch(lambda record: record["extra"].update(mem=memory_usage()))


def memory_usage():
    mu = 0.0 + resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # or ru_idrss ?
    for u in ["k", "m", "g", "t"]:
        if mu < 1000:
            if mu < 10: return "%3.1f"%mu + u
            return "%3.0f"%mu + u
        mu /= 1024