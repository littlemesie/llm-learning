# -*- coding:utf-8 -*-

"""
@date: 2022/8/9 上午10:10
@summary:
"""
import os
import sys
import logging.config
from core.system_config import project_dir


def configure_logging(path=None, level=logging.DEBUG, only_file=False):
    # 指定路径
    if path:
        log_file = os.path.join(project_dir, 'log/{}.log'.format(path))
    else:
        log_file = os.path.join(project_dir, 'log/root.log')

    formatter = '[%(asctime)s] - %(levelname)s: %(message)s'
    if only_file:
        logging.basicConfig(filename=log_file,
                            level=level,
                            format=formatter,
                            datefmt='%Y-%m-%d %H:%M:%S')
    else:
        logging.basicConfig(level=level,
                            format=formatter,
                            datefmt='%Y-%m-%d %H:%M:%S',
                            handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
                            )
