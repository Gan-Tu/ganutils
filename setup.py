from setuptools import setup, find_packages

def load_readme():
  return open("./README.md", "r").read()

setup(
  name = 'ganutils',
  packages = find_packages(),
  version = '0.12',
  license='MIT',
  description = 'This is an installable python package for scripts I wrote for myself.',
  long_description=load_readme(),
  long_description_content_type='text/markdown',
  author = 'Gan Tu',
  author_email = 'tugan0329@gmail.com',
  url = 'https://github.com/Michael-Tu/ganutils',
  keywords = ['tugan', 'utils', 'tools'],
  install_requires=[
      'torch',
      'torchvision',
      'numpy',
      'sklearn',
      'tqdm',
      'requests',
      'twilio',
      'pdoc3'
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
