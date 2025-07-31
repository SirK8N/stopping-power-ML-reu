#!/usr/bin/env python
# coding=utf-8
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
import os
os.environ["KERAS_BACKEND"] = "jax"
os.environ["JAX_ENABLE_X64"] = "True"
import functools
print = functools.partial(print, flush=True)

