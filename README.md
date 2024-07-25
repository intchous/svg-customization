# Text-Guided Vector Graphics Customization

## Installation

Create a new conda environment:

```shell
conda create --name svg_dream python=3.10
conda activate svg_dream
```

Install pytorch and the following libraries:

```shell
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install shapely 
pip install Pillow==9.5.0 scikit-image==0.19.3 opencv-python matplotlib
pip install numpy scipy timm scikit-fmm einops scikit-learn
pip install accelerate transformers safetensors datasets 
pip install cairosvg rembg pycpd easydict munch kornia
pip install faiss-cpu pytorch_metric_learning fast_pytorch_kmeans pypotrace
pip install --force-reinstall cython==0.29.36
pip install --no-build-isolation git+https://github.com/lucasb-eyer/pydensecrf.git
```

Install CLIP:

```shell
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

Install diffusers:

```shell
pip install diffusers
```

Install diffvg:

```shell
git clone https://github.com/BachiLi/diffvg.git
cd diffvg
git submodule update --init --recursive
conda install -y -c anaconda cmake
conda install -y -c conda-forge ffmpeg
pip install svgwrite svgpathtools cssutils torch-tools
python setup.py install
cd ..
cp -R ./upd_pydiffvg/* ~/anaconda3/envs/svg_dream/lib/python3.10/site-packages/diffvg-0.0.1-py3.10-linux-x86_64.egg/pydiffvg/
rm -rf pydiffvg
```

Install pypotrace:

```shell
sudo apt-get install build-essential python-dev libagg-dev libpotrace-dev pkg-config

git clone -b to_xml https://github.com/mehdidc/pypotrace.git
cd pypotrace
pip install .
cd ..
rm -rf pypotrace
```

## SD Finetuning

To train the model using the `dream_train.py` script, run:

```
cd dream
python dream_train.py --instance_img_dir 03670_animal_8
```

## Generating Customized Images

To generate customized images, run:

```
python dream_infer.py --concept_n 03670_animal_8 --prompt_description "wearing a top hat"
```

Note: Try different parameters for better results. Follow [custom-diffusion](https://github.com/adobe-research/custom-diffusion) to get better customizations.

## Exemplar SVG Preparation

1. Create a directory under `./test_svg_custom/`. For instance, you can create a directory named `test_animal`:

```
cd ..
mkdir ./test_svg_custom/test_animal
```

2. Place your exemplar SVG into the following directory:

```
./test_svg_custom/test_animal/svg_animal_ref/
```

3. Select suitable target images and move them to:

```
./test_svg_custom/test_animal/tar_animal_img/
```

Note: Target image's file name should contain exemplar SVG's file name.

## SVG Mask Processing

To process the exemplar SVG mask, use the following command:

```
python svg_mask_path_match.py --signature=animal
```

## Path Matching

Execute the following command to match the paths and get the initial SVG:

```
python img_path_match.py --signature=animal --is_segment=1 --is_mask_match=1
```

## Path Optimization

Download CLIP RN101 model:

```
mkdir -p ./models
wget -P ./models https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt
```

Run the optimization process with:

```
python svg_deform.py --signature=animal --losssign=procrustes
```

After optimizing, the results can be found in:

```python
./log/test_fig_deform_res/
or
./log/test_fig_deform/.../..._optm.svg
```
