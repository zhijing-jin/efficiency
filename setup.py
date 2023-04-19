from setuptools import setup, find_packages

with open('README.md') as f:
    long_description = f.read()

setup(
    name='efficiency',
    version='1.2',
    packages=find_packages(exclude=['tests*']),
    license='MIT',
    description='A package for efficient programming',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=['spacy', 'nltk', 'pandas', 'numpy'],
    url='https://github.com/zhijing-jin/efficiency',
    author='Zhijing Jin'
)
