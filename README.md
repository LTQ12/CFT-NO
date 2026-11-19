# CFT-NO: learning high-frequency scattering in co-moving Fock coordinates

Code for the paper

> T. Liu, L. Liu, C. Zhu, *Learning high-frequency scattering in co-moving Fock coordinates: frequency-independent parametric complexity and calibrated Bayesian shape inversion*, submitted to SIAM Journal on Scientific Computing.

The unknown is the Neumann trace of sound-soft scattering by a smooth convex obstacle in two dimensions. The trace is written on charts attached to the shadow points of each shape, graded on the Fock scale and carrying the incident and creeping-wave phases. On these charts, a small network corrects the analytic Fock solution. In these coordinates the parametric complexity of the shape-to-field map does not grow with the wavenumber.

This repository currently contains the core solver and surrogate, the GPU pipeline that generates the training data and trains and evaluates the models, the hybrid numerical-asymptotic and neural-operator baselines, and the scripts for the theory checks. The remaining experiment scripts (Bayesian inversion, a posteriori bounds, sample-complexity study) will be added.

## Layout

```
src/
  bie.py               Kress-Nystrom CFIE solver, far field (reference solutions)
  gpu_bie.py           the same Nystrom solver on the GPU (torch, optional cupy Bessel functions)
  fock.py              Fock functions F, G (tabulated in fock_table.npz)
  charts.py            arclength geometry, shadow points, co-moving chart layout
  g2_data.py           chart features and least-squares chart coefficients (training targets)
  mlp.py, surrogate.py chart-local network (numpy inference)
  fast_surrogate.py    fast evaluation of the surrogate: trace and far field for new shapes
  shapes.py            random convex star-shaped obstacles of the training family
  wide_data.py         generate all training data on the GPU (k from 32 to 2048)
  wide_train.py        train the chart-local network on the GPU
  wide_eval.py         far-field error of the surrogates for k = 32 ... 2048
  hna_galerkin.py      Galerkin HNA methods of Ecevit-Eruslu and Ecevit-Ozen (matrix-free on the GPU)
  band_eval.py         surrogate vs HNA methods vs Fock solution beyond the training band
  run_sota_bench_colab.py  FNO, TFNO and DeepONet baselines (neuraloperator library)
  bench_data_colab.py, bench_meta.py   test and benchmark sets generated from seeds
  g2_mlp_w256_local_ctx_hi.npz           trained model, k in [32, 512]
  g2_mlp_w256_local_ctx_hi_xhi_torch.npz trained model, k in [32, 2048]
theory/
  m1_parametric.py     Chebyshev mode counts along one-parameter shape families (numerical check of Section 3)
  m1_rho_fit.py        fitted Bernstein parameters
  fig_theory_data.py, make_fig_theory.py   fixed point vs co-moving points (theory figure of Section 3)
  fock_sector_check.py multiple-precision check of the Fock function in a sector (needs mpmath)
colab/README.md       how to run the GPU pipeline (Google Colab or any CUDA machine)
examples/quickstart.py surrogate vs Nystrom on one unseen obstacle
```

## Quick start

```
pip install numpy scipy matplotlib
python examples/quickstart.py 128
```

The script prints the relative far-field error of the surrogate against a Nystrom solution at 5.5 points per wavelength on an unseen star-shaped obstacle, for three incident directions. At k = 128 it is about 1e-5 to 4e-5.

## Reproducing the experiments

All data are generated from fixed random seeds; no data files need to be downloaded. The GPU pipeline (`colab/README.md`) regenerates the training sets, checks them against reference values of the original data, trains both models, and evaluates them from k = 32 to 2048. The same file lists the commands for the HNA and neural-operator comparisons.

## Contact

Lijun Liu (liulijun@xmu.edu.cn), Taiqian Liu (ltqworks123@163.com)

## License

MIT, see `LICENSE`.
