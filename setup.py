# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 10:00:42 2020

TRADE SECRET: CONFIDENTIAL AND PROPRIETARY INFORMATION.
Insight Sensing Corporation. All rights reserved.

@copyright: © Insight Sensing Corporation, 2020
@author: Tyler J. Nigon
@contributors: [Tyler J. Nigon]

This software contains confidential and proprietary information of Insight
Sensing Corporation and is protected by copyright, trade secret, and other
State and Federal laws. Its receipt or possession does not convey any rights to
reproduce, disclose its contents, or to manufacture, use or sell anything it
may describe. Reproduction, disclosure, or use without specific written
authorization of Insight Sensing Corporation is strictly forbidden.
"""

import setuptools

def readme():
    with open('README.md') as readme_file:
        return readme_file.read()

requirements = [
'pandas'
 ]

test_requirements = [
    # TODO: put package test requirements here
]

setuptools.setup(name='geoml',
                 version='0.0.1',
                 description=('Insight Sensing API for processing and '
                              'analizing research data.'),
                 long_description=readme(),
                 long_description_content_type="text/markdown",
                 url='https://github.com/insight-sensing/geoml',
                 author='Tyler J. Nigon',
                 author_email='tyler@insight-sensing.com',
                 copyright=('Copyright (c) Insight Sensing Corporation, 2020. '
                            'All rights reserved.'),
                 license=('TRADE SECRET: CONFIDENTIAL AND PROPRIETARY '
                          'INFORMATION.'),
                 packages=setuptools.find_packages(),
                 classifiers=[
                         'Development Status :: 4 - Beta',
                         'Intended Audience :: Internal/proprietary/confidential',
                         'Natural Language :: English',
                         'Operating System :: OS Independent',
                         'Programming Language :: Python',
                         ],
                 package_data={},
                 include_package_data=False,
                 install_requires=requirements,
#                 test_suite='geoml/tests',
                 tests_require=test_requirements,
                 zip_safe=False)
