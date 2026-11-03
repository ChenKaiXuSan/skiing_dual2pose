#!/usr/bin/env python

from pathlib import Path

from setuptools import find_namespace_packages, setup

PATH_ROOT = Path(__file__).parent


def load_requirements(path_dir=PATH_ROOT, comment_char="#"):
    requirements = []
    with open(path_dir / "requirements.txt", "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if comment_char in line:
                line = line[: line.index(comment_char)].strip()
            if line:
                requirements.append(line)
    return requirements


README = (PATH_ROOT / "README.md").read_text(encoding="utf-8")

setup(
    name="dual2pose",
    version="0.1.0",
    description="Canonical-aligned dual-view 3D pose fusion for skiing videos.",
    author="Kaixu Chen",
    author_email="chenkaixusan@gmail.com",
    url="https://github.com/ChenKaiXuSan/skiing_dual2pose",
    license="Apache-2.0",
    packages=find_namespace_packages(include=["dual2pose", "dual2pose.*", "configs", "configs.*", "cross_validation", "cross_validation.*", "pegasus", "pegasus.*"]),
    package_data={"configs": ["*.yaml", "model/*.yaml"], "pegasus.runtime": ["manifests/*.json", "requirements-*.txt", "*.sh"]},
    long_description=README,
    long_description_content_type="text/markdown",
    include_package_data=True,
    zip_safe=False,
    keywords=["deep learning", "pytorch", "3d pose", "multi-view"],
    python_requires=">=3.11",
    install_requires=load_requirements(PATH_ROOT),
    classifiers=[
        "Environment :: Console",
        "Natural Language :: English",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
    ],
)
