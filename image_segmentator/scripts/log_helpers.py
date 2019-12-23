"""Module to define function helpers to logging."""

import sys
from logging import Logger, getLogger, StreamHandler, INFO


def set_logger(mod_name: str) -> Logger:
    """
    Set the log to logging messages.

    Args:
        mod_name:
    """
    log = getLogger(mod_name)
    out_hdlr = StreamHandler(sys.stdout)
    out_hdlr.setLevel(INFO)
    log.addHandler(out_hdlr)
    log.setLevel(INFO)

    return log
