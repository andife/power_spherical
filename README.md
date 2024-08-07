<!--
SPDX-FileCopyrightText: 2020-2023 Nicola De Cao
SPDX-FileCopyrightText: 2024 Andreas Fehlner

SPDX-License-Identifier: MIT
-->

# The Power Spherical distribution

[![REUSE status](https://api.reuse.software/badge/github.com/andife/power_spherical/)](https://api.reuse.software/info/github.com/andife/power_spherical/)
[![SLSA 3](https://slsa.dev/images/gh-badge-level3.svg)](https://slsa.dev)

## Fork
*This fork was made to create a wheel for pypi (in the context of which the setup was also changed/updated to pyproject.toml and poetry) and to introduce good practices in [supply chain security](https://github.com/andife/power_spherical/blob/master/artifact-authenticity.md), such as sigstore and slsa.*

## Overview
This library contains a Pytorch implementation of the Power Spherical distribution, as presented in [[1]](#citation)(https://arxiv.org/abs/2006.04437).

## Dependencies

* **python>=3.9**
* **pytorch>=2.0**: https://pytorch.org

*Notice that older version could work but they were not tested.*

Optional dependency for [examples](https://github.com/andife/power_spherical/blob/master/example.ipynb) needed for plotting and numerical checks (again older version could work but they were not tested):
* **numpy>=1.18.1**: https://numpy.org
* **matplotlib>=3.1.1**: https://matplotlib.org
* **quadpy>=0.14.11**: https://pypi.org/project/quadpy (a paid license is needed)

## Installation

To install, run

```bash
pip install power-spherical
```

## Structure
* [distributions](https://github.com/andife/power_spherical/blob/master/power_spherical/distributions.py): Pytorch implementation of the Power Spherical and hyperspherical Uniform distributions. Both inherit from `torch.distributions.Distribution`.
* [examples](https://github.com/andife/power_spherical/blob/master/example.ipynb): Example code for using the library within a PyTorch project.

## Usage
Please have a look into the [examples](https://github.com/andife/power_spherical/blob/master/example.ipynb). We adapted our implementation to follow the structure of the [Pytorch probability distributions](https://pytorch.org/docs/stable/distributions.html).

Here a minimal example that demonstrate differentiable sampling:
```python
import torch
from power_spherical import PowerSpherical
p = PowerSpherical(
      loc=torch.tensor([0., 1.], requires_grad=True),
      scale=torch.tensor(4., requires_grad=True),
    )
p.rsample()
```




    tensor([-0.3820,  0.9242], grad_fn=<SubBackward0>)
    
and computing KL divergence with the uniform distribution:
```python
from power_spherical import HypersphericalUniform
q = HypersphericalUniform(dim=2)
torch.distributions.kl_divergence(p, q)
```




    tensor(1.2486, grad_fn=<AddBackward0>)

Examples of 2D and 3D plots are show in [examples](https://github.com/andife/power_spherical/blob/master/example.ipynb) and will generate something similar to these figures below.
<p align="center">
  <img class="paper_logo" src="https://i.imgur.com/4iITHS5.png" width=40%>
  <img class="paper_logo" src="https://i.imgur.com/zXZWr9H.png" width=40%>
</p>

Please cite [[1](#citation)] in your work when using this library in your experiments.

## Feedback
For questions and comments, feel free to contact [Nicola De Cao](mailto:nicola.decao@gmail.com).
For questions according to the package / setup / fork, feel free to contact [Andreas Fehlner](mailto:fehlner@arcor.de)

## License
MIT

## Citation
```
[1] De Cao, N., Aziz, W. (2020). 
The Power Spherical distrbution.
In Proceedings of the 37th International 
Conference on Machine Learning, INNF+.
```

BibTeX format:
```
@article{decao2020power,
  title={The Power Spherical distribution},
  author={
    De Cao, Nicola and
    Aziz, Wilker},
  journal={Proceedings of the 37th International Conference on Machine Learning, INNF+},
  year={2020}
}
```

