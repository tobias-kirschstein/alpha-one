from setuptools import setup

setup(name='alpha_one',
      version='0.1',
      description='AlphaOne',
      author='Tobias Kirschstein',
      author_email='kirschto@in.tum.de',
      packages=['alpha_one'],
      install_requires=['envyaml', 'trueskill', 'open_spiel', 'ray', 'numpy', 'tqdm'],
      zip_safe=False)
