from setuptools import setup, find_packages

setup(
    name='QuantaTextToSql',
    version='1.0',
    packages=find_packages(),
    description='Tools for use with the quanta text to sql project',
    author='Philip Quirke wt al',
    author_email='philipquirkenzgmail.com',
    install_requires=[
        'numpy>=1.18.1',
        'wheel'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)