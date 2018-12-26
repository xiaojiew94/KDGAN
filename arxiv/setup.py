from setuptools import setup, find_packages

# pip install -Ue .
setup(
    name='kdgan',
    author='Xiaojie Wang and Yixin Su',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'hello=scripts:hello',
            'aloha=scripts:aloha',
        ]
    },
)