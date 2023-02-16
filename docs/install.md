## Installation

### Requirements

- Linux
- Python 3.6+
- PyTorch 1.3+
- CUDA 9.2+
- GCC 5+
- [mmcv 0.6.2](https://github.com/open-mmlab/mmcv)

### Install VRDet

a. Create a conda virtual environment and activate it.

```shell
conda create -n vrdet python=3.7 -y
conda activate vrdet
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

```shell
conda install pytorch torchvision -c pytorch
```

Note: Make sure that your compilation CUDA version and runtime CUDA version match.
You can check the supported CUDA version for precompiled packages on the [PyTorch website](https://pytorch.org/).

`e.g.` If you have CUDA 10.1 installed under `/usr/local/cuda` and would like to install
PyTorch 1.6.0, you need to install the prebuilt PyTorch with CUDA 10.1.

```shell
conda install pytorch=1.6.0 cudatoolkit=10.1 torchvision=0.7.0 -c pytorch
```

`e.g.` If you have CUDA 9.2 installed under `/usr/local/cuda` and would like to install
PyTorch 1.3.1., you need to install the prebuilt PyTorch with CUDA 9.2.

```shell
conda install pytorch=1.3.1 cudatoolkit=9.2 torchvision=0.4.2 -c pytorch
```

Note: We use CUDA 9.2, pytorch 1.3.1 and torchvision 0.4.2 to successfully install the virtual environment. 
This set of versions can be used as a reference when you install Pytorch and torchvision.

c. Clone the VRDet repository.

```shell
git clone https://github.com/ShichengMiao16/VRDet.git --recursive
cd VRDet
```

d. Install build requirements and then install VRDet.

```shell
# install BboxToolkit
cd BboxToolkit
pip install -v -e .  # or "python setup.py develop"
cd ..

# install VRDet
pip install -r requirements/build.txt
pip install pycocotools
pip install mmcv==0.6.2
pip install -v -e .  # or "python setup.py develop"
```

Note:

1. The git commit id will be written to the version number with step d. The version will also be saved in trained models.
It is recommended that you run step d each time you pull some updates from github. If C++/CUDA codes are modified, then this step is compulsory.

    > Important: Be sure to remove the `./build` folder if you reinstall mmdet with a different CUDA/PyTorch version.

    ```shell
    pip uninstall mmdet
    rm -rf ./build
    find . -name "*.so" |xargs rm
    ```

2. Following the above instructions, VRDet is installed on `dev` mode, any local modifications made to the code will take effect without the need to reinstall it (unless you submit some commits and want to update the version number).

3. If you would like to use `opencv-python-headless` instead of `opencv-python`,
you can install it before installing MMCV.

4. Some dependencies are optional. Simply running `pip install -v -e .` will only install the minimum runtime requirements. To use optional dependencies like `albumentations` and `imagecorruptions` either install them manually with `pip install -r requirements/optional.txt` or specify desired extras when calling `pip` (e.g. `pip install -v -e .[optional]`). Valid keys for the extras field are: `all`, `tests`, `build`, and `optional`.

5. If you install torchvision < 0.5.0, please run `pip install "pillow<9"` to avoid `ImportError`.

### Install with CPU only
The code can be built for CPU only environment (where CUDA isn't available).

In CPU mode you can run the `$VRDet/demo/webcam_demo.py` for example.
However some functionality is gone in this mode:

- Deformable Convolution
- Deformable ROI pooling
- CARAFE: Content-Aware ReAssembly of FEatures
- nms_cuda
- sigmoid_focal_loss_cuda

So if you try to run inference with a model containing deformable convolution, you will get an error.
Note: We set `use_torchvision=True` on-the-fly in CPU mode for `RoIPool` and `RoIAlign`