# Copyright 2020-present, ai in motion
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
import io
import logging

from setuptools import (
    setup,
    find_packages
)
from pkg_resources import (
    DistributionNotFound,
    get_distribution, 
    parse_version,
)

logger = logging.getLogger()
logging.basicConfig(format='%(levelname)s - %(message)s')

def get_readme():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    with io.open(os.path.join(base_dir, "README.md"), encoding="utf-8") as f:
        return f.read()

def get_requirements():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(base_dir, "requirements.txt"), encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

def moai_info():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    version_file = os.path.join(current_dir, "moai", "__init__.py")
    with io.open(version_file, encoding="utf-8") as f:
        version = re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', f.read(), re.M).group(1)
        f.seek(0)
        author = re.search(r'^__author__ = [\'"]([^\'"]*)[\'"]', f.read(), re.M).group(1)
        f.seek(0)
        email = re.search(r'^__author_email__ = [\'"]([^\'"]*)[\'"]', f.read(), re.M).group(1)
        f.seek(0)
        licence = re.search(r'^__license__ = [\'"]([^\'"]*)[\'"]', f.read(), re.M).group(1)
        f.seek(0)
        url = re.search(r'^__homepage__ = [\'"]([^\'"]*)[\'"]', f.read(), re.M).group(1)
        f.seek(0)
        code_url = re.search(r'^__github_repo__ = [\'"]([^\'"]*)[\'"]', f.read(), re.M).group(1)
        f.seek(0)
        docs_url = re.search(r'^__documentation_page__ = [\'"]([^\'"]*)[\'"]', f.read(), re.M).group(1)
        f.seek(0)
        short_desc = re.search(r'^__docs__ = [\'"]([^\'"]*)[\'"]', f.read(), re.M).group(1)
        f.seek(0)
        keywords = re.search(r'^__keywords__ = [\'"]([^\'"]*)[\'"]', f.read(), re.M).group(1)
        return version, author, email, licence, \
            url, code_url, docs_url, \
            short_desc, keywords.split(',')

PACKAGE_NAME = 'moai-mdk'
VERSION, AUTHOR, EMAIL, LICENSE,\
URL, CODE_URL, DOCS_URL, \
DESCRIPTION, KEYWORDS = moai_info()

if __name__ == '__main__':
    logger.info(f"Installing {PACKAGE_NAME} (v: {VERSION}) ...")
    setup(
        name=PACKAGE_NAME,
        version=VERSION,
        author=AUTHOR,
        author_email=EMAIL,
        description=DESCRIPTION,
        long_description=get_readme(),
        long_description_content_type="text/markdown",
        keywords=KEYWORDS,
        licence_file='LICENCE',
        url=URL,
        project_urls={
            'Documentation': DOCS_URL,
            'Source': CODE_URL,
        },
        packages=find_packages(exclude=('docs', 'conf', 'outputs')),
        install_requires=get_requirements(),
        include_package_data=True,
        python_requires='~=3.7',
        entry_points={
            'console_scripts': [
                'moai=main:moai',              
            ],
        },
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "Programming Language :: Python :: 3.7",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Software Development :: Libraries",
            "Topic :: Software Development :: Libraries :: Python Modules",
        ],    
    )