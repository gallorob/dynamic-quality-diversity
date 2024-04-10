# Dynamic Quality-Diversity Search
Official repository for the "Dynamic Quality-Diversity Search" paper. Part of this work will be presented at GECCO 2024 as poster, full version of this work is available on [arXiv](https://arxiv.org/abs/2404.05769).

Abstract (from preprint version):
> Evolutionary search via the quality-diversity (QD) paradigm can discover highly performing solutions in different behavioural niches, showing considerable potential in complex real-world scenarios such as evolutionary robotics. Yet most QD methods only tackle static tasks that are fixed over time, which is rarely the case in the real world. Unlike noisy environments, where the fitness of an individual changes slightly at every evaluation, dynamic environments simulate tasks where external factors at unknown and irregular intervals alter the performance of the individual with a severity that is unknown a priori. Literature on optimisation in dynamic environments is extensive, yet such environments have not been explored in the context of QD search. This paper introduces a novel and generalisable Dynamic QD methodology that aims to keep the archive of past solutions updated in the case of environment changes. Secondly, we present a novel characterisation of dynamic environments that can be easily applied to well-known benchmarks, with minor interventions to move them from a static task to a dynamic one. Our Dynamic QD intervention is applied on MAP-Elites and CMA-ME, two powerful QD algorithms, and we test the dynamic variants on different dynamic tasks. 

## Project Description

When searching for solutions with any QD algorithm, we assume that the objective function and behavioural descriptions are fixed over time.

In this work we instead challenge this notion. If the environment in which we are evaluating solutions in changes over time (i.e.: is a dynamic environment), how do we ensure solutions we have found thus far are updated? Outdated solutions will hinder the search, as their properties (objective value and BCs) will be different.

<table>
	<tr>
		<td width=50%>
			<img src="/readme_resources/no_wind_turbulence.gif" title="no_wind_turbulence">
		</td>
		<td width=50%>
			<img src="/readme_resources/yes_wind_turbulence.gif" title="yes_wind_turbulence">
		</td>
	</tr>
	<tr>
		<td colspan=2>
			<b>Dynamic Lunar Lander:</b> The goal is to land the lander by controlling its thrusters, but dynamically changing the wind strength and turbulence can change the lander behavior significantly. Above, the same random solution simulated in the Lunar Lander environment before a shift in wind and turbulence (left) and after (right).
		</td>
	</tr>
</table>

Clearly, we cannot ignore the problem altogether, but we cannot re-evaluate all solutions found so far everytime we believe the environment has changed. In this work we instead argue that we can strike a balance between number of updated solutions we can keep and number of re-evaluations we must perform.

This new paradigm, which we call "Dynamic Quality-Diversity Search", requires little modifications to existing QD algorithms (here, we used MAP-Elites and CMA-ME as provided by the [pyribs](https://pyribs.org/) library). We identify three critical points in which we intervene: 1) Generation of offspring solutions; 2) Detection of changes in the environment; and 3) Re-evaluation of existing solutions.

From the experiments conducted on both the Dynamic Sphere and Dynamic Lunar Lander envrionments, we determined that detecting changes re-evaluating only oldest solutions and re-evaluating solutions that would be compared against offspring, usually leads to a desirable trade-off between accuracy and evaluations cost.

## Installing dependencies (the right way)
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

## Citing
Cite this work with
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
or with
```bibtex
@inproceedings{gallotta2024dynamic,
	author={Roberto Gallotta and Antonios Liapis and Georgios N. Yannakakis},
	title={Dynamic {Quality-Diversity} Search},
	booktitle={Proceedings of the Genetic and Evolutionary Computation Conference Companion},
	year={2024}
}
```