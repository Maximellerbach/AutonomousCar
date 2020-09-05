from setuptools import setup

setup(
    name='custom_modules',
    version='1.0',
    description='this is a python package with some custom modules',
    url='',
    author='Maxime Ellerbach',
    licence='MIT License',
    packages=['custom_modules'],
    install_requires=['opencv-python', 'pillow', 'numpy',
                      'pykalman', 'matplotlib', 'serial', 'tqdm']
)
