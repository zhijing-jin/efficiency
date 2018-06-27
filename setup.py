from setuptools import setup, find_packages

setup(
    name='efficiency',
    version='0.1',
    packages=find_packages(exclude=['tests*']),
    license='MIT',
    description='A package for efficient programming',
    long_description=open('README.md').read(),
    install_requires=['numpy'],
    url='https://github.com/zhijing-jin/efficiency',
    author='Z',
    author_email='zhijing@mit.edu'
)
