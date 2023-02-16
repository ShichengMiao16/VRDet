# BboxToolkit

BboxToolkit is a light codebase collecting some practical functions for oriented object detection.
The whole project is written in python, which can run in different platforms without compilation.

## Main Features

- **Various type of Bboxes**

    We define three different types of bounding boxes in BboxToolkit: horizontal bounding boxes (HBB), oriented bounding boxes (OBB), and 4 point polygon (POLY). Each type of boxes can convert to others easily.

- **Convinence for usage**

    The functions in BboxToolkit will decide the box type according to the input shape. There is no need to concern about the input box type.

## Installation

BboxToolkit requires following dependencies:

+ Python > 3
+ Numpy
+ Opencv-python
+ Shapely
+ Terminaltables
+ Pillow

BboxToolkit will automatically install dependencies when you install, so this section is mostly for your reference.

```shell
cd BboxToolkit
python setup.py develop # or "pip install -v -e ."
```

## Usage

Please refer to [USAGE.md](USAGE.md) for the basic usage.
