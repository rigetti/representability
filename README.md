>Notice: This is research code that will not necessarily be maintained to support further
>releases of Forest and other Rigetti Software. We welcome bug reports and PRs
>but make no guarantee about fixes or responses.

Representability
----------------

Representability is a library to work with linearly constrained mathematical programs over tensors.

It was designed as an easy to use framework for expressing semidefinite programs associated with varitional
2-RDM calculations.

At it's core, Representability is a collection of objects for Tensors and MultiTensor along with a methodology
for building dual basis elements (linear relationships between elements of the tensors).

Features of Tensor()
--------------------
* A general tensor object.  Holds data without having to subclass numpy array.

* Has an associated basis which is a bijection from indices to tensor indexing.  For example, a simple bijection might
be the the {index:index} map.  Another might be a {tuple: index} map where the tuple is some spin-adapted-geminal basis index.  This
is especially useful in the Fermionic case where the tensor objects are generally rank-4 tensors represented in matrix
form--e.g. mathematically indexed by geminals instead of monomials

* The Tensor object is iterable and returns the basis indexing

* Tensor can be easily "indexed" into by returning the vectorized variable number.  For example the matrix
{{0, 1}, {2, 3}} vectorized in C-ordering turns into a vector {0, 1, 2, 3}.  index_vectorized(1, 0) will return the
vectorized index '2'.  This is useful when defining a vector space of an aggregate of Tensor objects.  When a basis is
defined on the Tensor index_vectorized uses the basis.rev() to get the position in a symmetrized form.

Features of MultiTensor()
------------------------

* As the name suggests this is an object that aggregates Tensor objects together to define a vector space.

* Vectorization is performed in the canonical way

* You can associate a DualBasis to a MultiTensor objects and synthesize the linear operator on the vector space as
sparse operators.

Features of DualBasis()
-----------------------

* Container for DualBasisElements.

* Can be added together to create a larger set of operators.

Feature of DualBasisElement()
-----------------------------

* Container for a single dual basis element.

* Each element can be extended and modified

* Simplification can be performed to avoid rewriting and minimizing if-else statements in the code.

RDM Projection
--------------

* Higham purification finds the closest positive semidefinite matrix with a fixed trace.  This method is efficient
because it does not require solving an SDP with a trace constraint.  It involves a matrix diagonalization and a
root finding step of monotonic function of the eigenvalues.

* iterative 2-RDM purification  Iteratively purify a 2-RDM by Higham purification then mapping to
other marginals and performing the Higham purification on those states.

* SDP 2-RDM purification. The method that finds the closest 2-RDM to a measured 2-RDM according the the rules of
2-positivity.

Information on all these techniques can be found in DOI:https://doi.org/10.1088/1367-2630/aab919


## Installation

We recommend installing representability in a fresh environment built with a base python 3.6 installation.
This can be accomplished with `conda` through the following commands

```
conda create -n rep_env python=3.6
source activate rep_env
```

You can install representability directly from the Python package manager `pip` using:
```
pip install -r requirements.txt
pip install representability -e .
```

The `-e .` reference the python package manager to the local directory for library source-code.

Current Roadmap
---------------

* Finish the Fermion module: This module performs variational 2-RDM code and the marginal reconstruction code for the
measurement project.  Need to implement the complete spin-adapted version and with T1, T2 T2', and sharp N-rep
conditions.

* qubit module: Tomography from shot data or purification of qubit density matrices.  Approximate reconstruction.

* Unify the density *_maps.py with Tensor objects and the appropriate dual basis object.  DualBasis was designed
to be general enough to perform this mapping.

* Boson representability

* Majorana representability


## How to cite the representability

If you use the `representability` please cite the repository as follows:

bibTex:
```
@misc{rep2018.0.0.1,
  author = {Nicholas Rubin,
  title = {Representability},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/rigetticomputing},
  commit = {the commit you used}
}
```
