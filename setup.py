from setuptools import setup
import os
from glob import glob

package_name = 'astar_navigation'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml') + glob('config/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Aun',
    maintainer_email='you@example.com',
    description='A* path planning and pure-pursuit navigation for ROS 2 Humble',
    license='MIT',
    entry_points={
        'console_scripts': [
            'astar_node = astar_navigation.astar_node:main',
        ],
    },
)
