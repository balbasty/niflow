from setuptools import setup

setup(
    name='niflow',
    version='0.1.0',
    packages=['niflow'],
    url='https://github.com/balbasty/niflow',
    license='MIT',
    author='Yael Balbastre',
    author_email='yael.balbastre@gmail.com',
    description='Neuroimaging in tensorflow',
    python_requires='>=3.5',
    install_requires=['unik', 'nibabel'],
    dependency_links=['https://github.com/balbasty/unik/tarball/master#egg=unik-0.1.0'],
)
