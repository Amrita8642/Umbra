from setuptools import setup, find_packages

setup(
    name="umbra-shadowworld",
    version="0.1.0",
    description="UMBRA - ShadowWorld Meta Environment",
    packages=find_packages(),
    install_requires=[
        "gymnasium>=0.29.0",
        "numpy",
        "transformers",
        "torch",
        "peft",
        "trl",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0"
    ],
)