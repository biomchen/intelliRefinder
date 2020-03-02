from setuptools import setup

install_reqs = open('requirements.txt')
lines = install_reqs.readlines()
reqs = [str(each_req) for each_req in lines]

setup(name='intelliRefinder',
      version='0.1',
      description='Using ML to identify mortgage refinance opportunities',
      long_description = read_file('README.md'),
      author='Meng Chen',
      author_email='meng.chen03@gmail.com',
      url='https://github.com/biomchen/intelliRefinder',
      license='MIT',
      install_requires=reqs)
