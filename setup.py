from setuptools import setup
from setuptools import find_packages

long_description = '''
Batch generation framework for Keras designed as a convenient and flexible tool
for making keras-ready batch generators from pandas DataFrames and a bunch of
sklearn transformers of your choice 
'''

setup(name='keras-batchflow',
      use_scm_version=True,
      description='Batch generation framework for Keras',
      long_description=long_description,
      author='Maxim Scherbak',
      author_email='maxim.scherbak@gmail.com',
      url='https://github.com/maxsch3/batchflow',
      download_url='https://github.com/maxsch3/batchflow',
      license='MIT',
      setup_requires=['setuptools_scm'],
      install_requires=['numpy>=1.9.1',
                        'scipy>=0.14',
                        'scikit-learn',
                        'pandas'],
      extras_require={
          'visualize': ['pydot>=1.2.4'],
          'tests': ['pytest',
                    'markdown'],
      },
      classifiers=[
          'Development Status :: 2 - Pre-Alpha',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules'
      ],
      packages=find_packages())