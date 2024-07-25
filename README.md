<img height="200" src="https://raw.githubusercontent.com/WayScience/coSMicQC/main/logo/with-text-for-light-bg.png?raw=true">

# Single cell Morphology Quality Control

[![Build Status](https://github.com/WayScience/coSMicQC/actions/workflows/run-tests.yml/badge.svg?branch=main)](https://github.com/WayScience/coSMicQC/actions/workflows/run-tests.yml?query=branch%3Amain)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

> :stars: Navigate the cosmos of single-cell morphology with confidence — coSMicQC keeps your data on course!

coSMicQC is a software tool to evaluate converted single-cell morphology outputs from CytoTable.

When performing object detection in CellProfiler, it is very hard to get "perfect" compartment segmentation across a large high-throughput screen.
Technical artifacts can occur during segmentation, including under or over segmentation or segmentation of background or smudges/bright artifacts.

In single-cell analysis, interesting phenotypes often emerge by looking at the morphological differences. However, the presence of technical outliers in the data can obscure these biological insights, undermining the validity of the findings.

By utilizing specific CellProfiler extracted morphological features, which we show work best with AreaShape features, you can detect technically incorrect segmentations that you can either label or remove prior to further preprocessing steps (like with pycytominer).

## Installation

Install coSMicQC from [PyPI](https://pypi.org/project/coSMicQC/) or from source:

```shell
# install from pypi
pip install coSMicQC

# install directly from source
pip install git+https://github.com/WayScience/coSMicQC.git
```

## Contributing, Development, and Testing

Please see [contributing.md](docs/source/contributing.md) for more details on contributions, development, and testing.

## References

- [pycytominer](https://github.com/cytomining/pycytominer)
- [CellProfiler](https://github.com/CellProfiler/CellProfiler)
- [CytoTable](https://github.com/cytomining/CytoTable)
