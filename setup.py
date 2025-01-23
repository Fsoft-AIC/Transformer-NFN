from setuptools import find_packages, setup

setup(
    name='transformer-nfn',
    version='0.1.2',
    description='Transformer Neural Functional Network (NFN) layers',
    url='',
    author='',
    author_email='',
    license='MIT',
    packages=find_packages(exclude=["examples", "experiments", "tests"]),
    install_requires=['einops', 'torch'],
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
    ]
)
