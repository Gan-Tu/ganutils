from setuptools import setup
setup(
  name = 'ganutils',
  packages = ['ganutils'],
  version = '0.8',
  license='MIT',
  description = 'This is an installable python package for scripts I wrote for myself.',
  author = 'Gan Tu',
  author_email = 'tugan0329@gmail.com',
  url = 'https://github.com/Michael-Tu/ganutils',
  keywords = ['tugan', 'utils', 'tools'],
  install_requires=[
      'torch',
      'numpy',
      'sklearn',
      'tqdm',
      'requests'
  ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
  ],
)
