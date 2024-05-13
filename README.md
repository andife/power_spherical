# The Power Spherical distribution

## Fork
*This fork was made to create a wheel for pypi (in the context of which the setup was also changed/updated to pyproject.toml and poetry).*

## Overview
This library contains a Pytorch implementation of the Power Spherical distribution, as presented in [[1]](#citation)(https://arxiv.org/abs/2006.04437).

## Dependencies

* **python>=3.9**
* **pytorch>=2.0**: https://pytorch.org

*Notice that older version could work but they were not tested.*

Optional dependency for [examples](https://github.com/andife/power_spherical/blob/master/example.ipynb) needed for plotting and numerical checks (again older version could work but they were not tested):
* **numpy>=1.18.1**: https://numpy.org
* **matplotlib>=3.1.1**: https://matplotlib.org
* **quadpy>=0.14.11**: https://pypi.org/project/quadpy

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

Examples of 2D and 3D plots are show in [examples](https://github.com/nicola-decao/power_spherical/blob/master/example.ipynb) and will generate something similar to these figures below.
<p align="center">
  <img class="paper_logo" src="https://i.imgur.com/4iITHS5.png" width=40%>
  <img class="paper_logo" src="https://i.imgur.com/zXZWr9H.png" width=40%>
</p>

Please cite [[1](#citation)] in your work when using this library in your experiments.

## Feedback
For questions and comments, feel free to contact [Nicola De Cao](mailto:nicola.decao@gmail.com).

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
  title={The Power Spherical distrbution},
  author={
    De Cao, Nicola and
    Aziz, Wilker},
  journal={Proceedings of the 37th International Conference on Machine Learning, INNF+},
  year={2020}
}
```


## Release artifact authenticity
*The page is heavily adapted from https://github.com/MatthiasValvekens/pyHanko/blob/master/docs/artifact-authenticity.rst*

### Overview ###

power-spherical uses several mechanisms to provide assurances regarding the authenticity of
its release artifacts, with the goal of mitigating its exposure to downstream software
supply chain issues.

For the purposes of all security checks described here GitHub effectively acts as the trust root.
    

### Sigstore signatures ###

#### Scope ####

power-spherical's release pipeline in GitHub Actions uses the Github Actions OIDC token
to certify release artifacts as originating from a specific repository or ref.
It does so using the public `Sigstore <https://sigstore.dev>`_ instance.
The following are important to keep in mind:

 * The machine identity used to obtain the Sigstore signature is also the one
   used to authenticate to PyPI.
 * While Sigstore's OIDC-based keyless signing procedure does not rely on any
   maintainer-controlled secrets, deploying cannot be done without manual
   maintainer review, and only repository admins can push ``v*`` tags.

Long story short, as long as you trust GitHub's security controls, these checks
are appropriate.


#### Verifying Sigstore signatures issued through GitHub Actions OIDC ####

 #. Install ``sigstore``
 #. Download the ``.sigstore`` bundles from the GitHub release page
 #. Download the release artifacts you are interested in through whichever channel you prefer
    (e.g. using ``pip wheel``, or manual download from GitHub/PyPI)

.. code-block:: bash

    #!/bin/bash

    export EXPECTED_VERSION=<version number goes here>
    export REPO=andife/power-spherical
    sigstore verify github \
        --cert-identity "https://github.com/$REPO/.github/workflows/release.yml@refs/tags/v$EXPECTED_VERSION" \
        --ref "refs/tags/v$EXPECTED_VERSION" \
        --repo "$REPO" \
        power-spherical-$EXPECTED_VERSION-*.whl power-spherical-$EXPECTED_VERSION.tar.gz

### SLSA provenance data ###

#### Scope ####

The idea behind supplying SLSA provenance data is to allow people to validate that
a given artifact was built using the expected parameters on some pre-agreed
build platform (in casu GitHub Actions).

The SLSA provenance data is also backed by Sigstore.

  * PyPI does not integrate SLSA support natively,
    so the provenance data is only added to GitHub releases and will not be automatically
    checked by your package manager (e.g. ``pip``).
    Also, power-spherical's SLSA scope does **not** include any guarantees about transitive dependencies
    that your package manager may or may not pull in.


  * The security guarantees of this process largely coincide with those of the
    Sigstore-based signatures from the previous section, but the packaging/tooling
    is slightly different.
    Until the Python ecosystem integrates SLSA more closely, either mechanism
    gets you pretty much the same thing if you validate using the methods
    described on this page. Of course, YMMV if you apply additional controls on the
    authenticated metadata.


#### Verifying SLSA provenance data on release builds ####

power-spherical will include `SLSA provenance data <https://slsa.dev/>`_.
To verify one or more power-spherical release artifacts, perform the following steps:

 #. Install ``slsa-verifier``
 #. Download the ``multiple.intoto.json`` provenance file from the GitHub release page
 #. Download the release artifacts you are interested in through whichever channel you prefer
    (e.g. using ``pip wheel``, or manual download from GitHub/PyPI)
 #. Run the snippet below.


.. code-block:: bash

    export EXPECTED_VERSION=<version number goes here>
    export REPO=andife/power-spherical
    slsa-verifier verify-artifact \
        --source-tag "v$EXPECTED_VERSION" \
        --provenance-path ./multiple.intoto.jsonl \
        --source-uri "github.com/$REPO" \
        pyHanko-$EXPECTED_VERSION-*.whl pyHanko-$EXPECTED_VERSION.tar.gz

You can of course inspect the validated provenance data for any other authenticated metadata
that you think might be useful.
