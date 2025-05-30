from setuptools import setup
import os
from glob import glob

package_name = 'pa3_planner'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Rohan Ray',
    maintainer_email='rohan.ray@example.com',
    description='ROSbot movement planner for PA3',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'planner_node = pa3_planner.planner_node:main',
        ],
    },
) 