# from setuptools import setup, find_packages

# setup(
#     name="graylog",
#     version="0.0.1",
#     packages=find_packages(),
#     package_data={'graylog': ['conf/**/*.yaml']},
#     include_package_data=True,
#     # entry_points={
#     #     "console_scripts": [
#     #         "hydra-graylog=hydra_graylog.graylog:hydra_main",
#     #     ],
#     # },
#     install_requires=[
#         "hydra-core==1.0.3",
#         "graypy>=2.1.0",
#     ],
#     author="Georgios Albanis",
#     author_email="giorgos@moverse.ai",
#     description="Hydra plugin for logging to Graylog",
#     license="MIT",
#     keywords="hydra graylog",
#     url="https://github.com/ai-in-motion/moai",
#     classifiers=[
#         "Development Status :: 4 - Beta",
#         "Programming Language :: Python :: 3.9",
#         "Topic :: Utilities",
#     ],
# )

import os
import io
import logging

from setuptools import (
    setup,
    find_namespace_packages
)

logger = logging.getLogger()
logging.basicConfig(format='%(levelname)s - %(message)s')

def get_readme():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    with io.open(os.path.join(base_dir, "README.md"), encoding="utf-8") as f:
        return f.read()

PACKAGE_NAME = 'graylog'
VERSION = 1.0
AUTHOR = 'Georgios Albanis'
EMAIL = 'giorgos@moverse.ai'

def get_requirements():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(base_dir, "requirements.txt"), encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

if __name__ == '__main__':
    logger.info(f"Installing {PACKAGE_NAME} (v: {VERSION}) ...")
    setup(
        name=PACKAGE_NAME,
        version=VERSION,
        author=AUTHOR,
        author_email=EMAIL,
        description='the `graylog` package plugin for logging to Graylog',
        long_description=get_readme(),
        long_description_content_type="text/markdown",
        keywords="hydra graylog",
        licence_file='LICENCE',
        url='',
        project_urls={
            'Documentation': '',
            'Source': '',
        },
        packages=find_namespace_packages(
            # include=["hydra_plugins.*", "bdtk"],
            exclude=('docs', 'outputs', 'test_data', 'data', 'scripts', 'tests')
        ),
        package_data={'graylog': ['conf/**/*.yaml']},
        include_package_data=True,
        install_requires=get_requirements(),
        python_requires='~=3.7',                
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