from setuptools import setup, find_packages

setup(
    name="intr-paquetlab",
    version="0.1.0",
    author="Your Name",
    description="A package for intr-paquetlab experiments",
    packages=find_packages(include=["intr", "intr.*"]),
    include_package_data=True,
    install_requires=[
        # optionally parse from requirements.txt
        line.strip() for line in open("requirements.txt").readlines() if line.strip() and not line.startswith("#")
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # or update per your LICENSE
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
