from distutils.core import setup
from glob import glob

scripts = glob('bin/*')
scripts = [s for s in scripts if '~' not in s]

setup(
    name='shredx',
    version='v0.9.0',
    description=('Run the shredder image deblender on '
                 'images processed with sextractor'),
    license='GPL',
    author='Erin Scott Sheldon',
    author_email='erin.sheldon@gmail.com',
    packages=['shredx'],
    scripts=scripts,
)
