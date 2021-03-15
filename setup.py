from setuptools import setup, find_packages

setup(
    name='covidnlp',
    version='1.0',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'matplotlib',
        'multiprocess',
        'pandas',
        'psutil',
        'pyrogue',
        'tensorboard',
        'tensorboardX',
        'tensorflow',
        'torch==1.7.1',
        'transformers',
        'tqdm',
        'seaborn',
        'sentence-transformers',
    ],
)
