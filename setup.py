from setuptools import setup

VERSION = "0.3.0"
DESCRIPTION = "GTSF Quant Sector shared data and feature library."

setup(
    name="data_universe",
    version=VERSION,
    description=DESCRIPTION,
    packages=[
        "data_universe",
        "data_universe.sources",
        "data_universe.options",
        "data_universe.features",
        "data_universe.labels",
        "data_universe.events",
    ],
    install_requires=[
        "pandas",
        "numpy",
        "requests",
        "pyarrow",
        "scipy",
        "pyyaml",
        "lxml",
    ],
    extras_require={
        "sec": [
            "sec_parser @ git+https://github.com/GTSF-Quantitative-Sector/sec_parser.git",
        ],
        "s3": ["boto3"],
        "dev": ["pytest", "ruff"],
    },
    python_requires=">=3.9",
)
