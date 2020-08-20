from setuptools import setup

# Note
# ----
# Git dependencies are not handled the same way by setuptools and pip:
# - setuptools browses repositories listed in `dependency_links`
# - pip uses the syntax 'repo @ url' in `install_requires`
# I am using both, because I want users to be able to choose between
# setuptools and pip.

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
    install_requires=['unik @ git+https://github.com/balbasty/unik#egg=unik-0.1.0',
                      'nibabel'],
    dependency_links=['git+https://github.com/balbasty/unik#egg=unik-0.1.0'],
)
