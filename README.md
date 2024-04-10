# Dynamic Quality-Diversity Search

Official repository for the "Dynamic Quality-Diversity Search" paper. Part of this work will be presented at GECCO 2024 as poster, full version of this work is available on [arXiv](https://arxiv.org/abs/2404.05769).

### Installing dependencies (the right way)
```shell
conda install -c conda-forge pyribs-visualize
conda install scikit-learn-intelex
```

For Gymnasium environments:
```shell
conda install -c anaconda numpy swig
conda install -c conda-forge pybox2d gymnasium[box2d] moviepy dask distributed pygame
```

On Windows, make sure the following Visual Studio Build Tools are installed in order for Box2D to work properly:
- Desktop Development with C++
  - C++ Build Tools core features
  - C++ 2022 Redistributable Update
  - C++ core desktop features
  - MSVC v143 - VS 2022 C++ x64/x86 build tools (latest)
  - Windows 11 SDK (10.0.22621.0)

For interactive plotting and more, install Jupyter:
```shell
conda install -c anaconda jupyter
```
If it doesn't work, install it with `pip`:
```shell
pip install jupyter
```

We use `EntropyHub` to measure entropy in `dyn_env_explorer.py`, so make sure it's installed:
```shell
pip install entropyhub
```

Remember to launch with
```shell
python -m sklearnex my_application.py
```
to leverage the acceleration library for `scikit-learn`.

### References
Cite this work as
```bibtex
@misc{gallotta2024dynamic,
      title={Dynamic Quality-Diversity Search}, 
      author={Roberto Gallotta and Antonios Liapis and Georgios N. Yannakakis},
      year={2024},
      eprint={2404.05769},
      archivePrefix={arXiv},
      primaryClass={cs.NE}
}
```