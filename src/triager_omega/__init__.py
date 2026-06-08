"""Triager-Omega — triaje automático de bugs sobre Mozilla BugsRepo."""

import sys

from loguru import logger

__version__ = "0.1.0"

# Sink por defecto a INFO: oculta el ruido DEBUG de carga pero conserva el
# reporte de los módulos. Configurable con TRIAGER_LOG_LEVEL.
import os as _os

logger.remove()
logger.add(sys.stderr, level=_os.getenv("TRIAGER_LOG_LEVEL", "INFO"))
