from setuptools import setup

setup(
   name='control_gen',
   version='1.0.0',
   author='Jambay Kinley',
   author_email='jambay606@gmail.com',
   packages=['control_gen'],
   url='https://github.com/JKinx/nlp-viz',
   install_requires=[
       "torch",
       "nltk",
       "py-rouge",
       "tensorboardX",
       "num2word"
   ],
)