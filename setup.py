from setuptools import setup, find_packages


install_requires = [
    "gym>=0.15,<=0.23",
]


extras_require = {

}


setup(
    name="trust",
    description="A Reinforcement Learning library in jax that works.",
    author="Learning and Signal Processing Group",
    author_email="e.pignatelli@ucl.ac.uk",
    url="https://github.com/lasp-ucl/trust",
    packages=find_packages(),
    install_requires=install_requires,
    extras_require=extras_require,
    python_requires=">=3.9",
)
