from setuptools import setup, find_packages


setup(
    name="trust",
    description="A Reinforcement Learning library in jax that works.",
    author="Learning and Signal Processing Group",
    author_email="e.pignatelli@ucl.ac.uk",
    url="https://github.com/lasp-ucl/trust",
    packages=find_packages(),
    python_requires=">=3.9",
)
