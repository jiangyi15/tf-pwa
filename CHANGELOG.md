# Changelog

## [Unreleased](https://github.com/jiangyi15/tf-pwa/tree/HEAD)

[Full Changelog](https://github.com/jiangyi15/tf-pwa/compare/v0.2.1...HEAD)

## [v0.2.1](https://github.com/jiangyi15/tf-pwa/tree/v0.2.1) (2024-02-10)

**Irreducible Tensor Formula**

[Covariant orbital-spin scheme for any spin based on irreducible tensor](https://inspirehep.net/literature/2620146).
[#138](https://github.com/jiangyi15/tf-pwa/pull/138)

**Added**

- `no_q0=True` options to remove `q0` dependences of barrier factor.
  [#132](https://github.com/jiangyi15/tf-pwa/pull/132)

- `ls_selector` options for redundant degree of freedom when decay to massless
  particle ($\gamma$). [#138](https://github.com/jiangyi15/tf-pwa/pull/138)

- Square Dalitz plot generator.
  [#136](https://github.com/jiangyi15/tf-pwa/pull/136)

- New models.

  - `linear_npy` and `linear_txt`
    [#137](https://github.com/jiangyi15/tf-pwa/pull/137)

  - `FlatteGen` and `Flatte2`
    [#132](https://github.com/jiangyi15/tf-pwa/pull/132)

- More plot functions and options.

  - `config.plot_partial_wave_interf`
    [#137](https://github.com/jiangyi15/tf-pwa/pull/137)

  - `partial_waves_function`
    [#135](https://github.com/jiangyi15/tf-pwa/pull/135)

  - `MuliConfig.plot_partial_wave`
    [#133](https://github.com/jiangyi15/tf-pwa/pull/133)

  - `force_legend_labels`, `add_chi2`
    [#132](https://github.com/jiangyi15/tf-pwa/pull/132)

- More nll models. `constr_frac` model support constrians on fractions.
  [#128](https://github.com/jiangyi15/tf-pwa/pull/128)
  [#129](https://github.com/jiangyi15/tf-pwa/pull/129)

**Changed**

- `no_q0` options to remove `q0` dependences of barrier factor.

- More options for `solve_pole`.
  [#134](https://github.com/jiangyi15/tf-pwa/pull/134)

- Update documents.
  [#130](https://github.com/jiangyi15/tf-pwa/pull/130)[#131](https://github.com/jiangyi15/tf-pwa/pull/131)[#135](https://github.com/jiangyi15/tf-pwa/pull/135)

## [v0.2.0](https://github.com/jiangyi15/tf-pwa/tree/v0.2.0) (2023-11-01)

[Full Changelog](https://github.com/jiangyi15/tf-pwa/compare/v0.1.7...v0.2.0)

**Better support for large data**

[#88](https://github.com/jiangyi15/tf-pwa/pull/88)[#94](https://github.com/jiangyi15/tf-pwa/pull/94)[#95](https://github.com/jiangyi15/tf-pwa/pull/95)[#100](https://github.com/jiangyi15/tf-pwa/pull/100)[#105](https://github.com/jiangyi15/tf-pwa/pull/105)[#110](https://github.com/jiangyi15/tf-pwa/pull/110)[#114](https://github.com/jiangyi15/tf-pwa/pull/114)[#115](https://github.com/jiangyi15/tf-pwa/pull/115)

- Preprocessor and Amp Model. Support different way of amplitude calculation
  and caching.

|              | preprocessor                      | amp_model                      |
| ------------ | --------------------------------- | ------------------------------ |
| default      | p4 -> m + angle                   | $a_i f_i(m) D_i(angle)$        |
| cached_amp   | p4 -> m + angle + $D_i$           | $a_i f_i(m) D_i$               |
| cached_shape | p4 -> m + angle + $f_i D_i + D_j$ | $a_i f_i D_i + a_j f_j(m) D_j$ |
| p4_directly  | p4                                | $a_i f_i(m(p4))D_i(angle(p4))$ |

$a_i$ is the fit parameters. $f_i(m)$ is the mass dependent parts. $D_i(angle)$
is the angle parts.

- Use Tensorflow Dataset for `lazy_call: True`. More options for caching.
  Support caching in CPU and Disk(File).

Requirement of memory (units: $10^7 \times 16\approx 1.2$GB）

| data          | size                                              |
| ------------- | ------------------------------------------------- |
| p4            | 4 N(event) N(particles)                           |
| m             | N(event) N(particles)                             |
| angle         | N(event) N(topo chains)(9 N(final particles)-6)   |
| $f_iD_i, D_i$ | N(event) N(helicity combination) N(partial waves) |

**Better support AD**

- AD for solving pole. [#83](https://github.com/jiangyi15/tf-pwa/pull/83)

- AD when changing parameters value.
  [#91](https://github.com/jiangyi15/tf-pwa/pull/91)

- Systematic uncertanties for fixed value.
  [#111](https://github.com/jiangyi15/tf-pwa/pull/111)

**<font color=red>Numeric</font>**

- Exchange helicity of identical particles when $J\neq0$.
  [#85](https://github.com/jiangyi15/tf-pwa/pull/85)

- Fixed some error when m=0 in generating PHSP sample.
  [#123](https://github.com/jiangyi15/tf-pwa/pull/123)

**<font color=green>Validation</font>**

Chains fraction in 3-body and 4-body:
[#82](https://github.com/jiangyi15/tf-pwa/pull/82)

**Added**

- More models.

  - kmatrix simple: [#80](https://github.com/jiangyi15/tf-pwa/pull/80)
  - sppchip: [#121](https://github.com/jiangyi15/tf-pwa/pull/121)
  - Multi BW: [#122](https://github.com/jiangyi15/tf-pwa/pull/122)

- Support constrains of variables via transform `from_trans`.
  [#112](https://github.com/jiangyi15/tf-pwa/pull/112)

- Some useful scripts.

  - 3d angle: [#96](https://github.com/jiangyi15/tf-pwa/pull/96)
  - Check interf: [#99](https://github.com/jiangyi15/tf-pwa/pull/99)
  - Check nan: [#116](https://github.com/jiangyi15/tf-pwa/pull/116)

- More options for input data.

  - extra_var: [#87](https://github.com/jiangyi15/tf-pwa/pull/87)
  - Root files: [#101](https://github.com/jiangyi15/tf-pwa/pull/101)
  - weight_smear: [#124](https://github.com/jiangyi15/tf-pwa/pull/124)

- Experimental Multi GPU support.

  - Multi data sample: [#93](https://github.com/jiangyi15/tf-pwa/pull/93)
  - Batching: [#126](https://github.com/jiangyi15/tf-pwa/pull/126)

- Better support for resolutions.
  [#81](https://github.com/jiangyi15/tf-pwa/pull/81)
  [#87](https://github.com/jiangyi15/tf-pwa/pull/87)
  [#92](https://github.com/jiangyi15/tf-pwa/pull/92)
  [#108](https://github.com/jiangyi15/tf-pwa/pull/108)

- More plot options.

  - ref_amp: [#84](https://github.com/jiangyi15/tf-pwa/pull/84)
  - legend_outside: [#102](https://github.com/jiangyi15/tf-pwa/pull/102)
    [#103](https://github.com/jiangyi15/tf-pwa/pull/103)
    [#109](https://github.com/jiangyi15/tf-pwa/pull/109)
  - 2d pull: [#104](https://github.com/jiangyi15/tf-pwa/pull/104)

- Print as roofit. [#77](https://github.com/jiangyi15/tf-pwa/pull/77)

- `grad_scale` for fit. [#97](https://github.com/jiangyi15/tf-pwa/pull/97)

- Symbolic formula for LS to Helicity.
  [#98](https://github.com/jiangyi15/tf-pwa/pull/98)

- config for d in barrier factor.
  [#106](https://github.com/jiangyi15/tf-pwa/pull/106)

- Extended likelihood for base nll model.
  [#113](https://github.com/jiangyi15/tf-pwa/pull/113)

- Support custom NLL model.
  [#120](https://github.com/jiangyi15/tf-pwa/pull/120)

**Changed**

- Fixed res in plot_partial_wave.
  [#90](https://github.com/jiangyi15/tf-pwa/pull/90)

- Avoid same name in one decay chain.
  [#106](https://github.com/jiangyi15/tf-pwa/pull/106)

- Remove decay together when in Particle.decays.
  [#119](https://github.com/jiangyi15/tf-pwa/pull/119)

- Update of document. [#78](https://github.com/jiangyi15/tf-pwa/pull/78)
  [#107](https://github.com/jiangyi15/tf-pwa/pull/107)
  [#117](https://github.com/jiangyi15/tf-pwa/pull/117)
  [#118](https://github.com/jiangyi15/tf-pwa/pull/118)
  [#125](https://github.com/jiangyi15/tf-pwa/pull/125)

## [v0.1.7](https://github.com/jiangyi15/tf-pwa/tree/v0.1.7) (2022-07-06)

[Full Changelog](https://github.com/jiangyi15/tf-pwa/compare/v0.1.6...v0.1.7)

**<font color=red>Numeric</font>**

- Fixed wrong phase space shape when N >= 4.
  [#76](https://github.com/jiangyi15/tf-pwa/pull/76)
- Fiexd `GS_rho` model bug. [#67](https://github.com/jiangyi15/tf-pwa/pull/67)
- Fixed a small bug in L=4 blatt weisskoft coefficient.
  [#61](https://github.com/jiangyi15/tf-pwa/pull/61)

**Added**

- Some importance sampling method.
  [#76](https://github.com/jiangyi15/tf-pwa/pull/76)
- `lazy_call: True` option. [#74](https://github.com/jiangyi15/tf-pwa/pull/74)
- `cp_particles` option for self CP symmetry decay such as
  `psi -> pi+ pi- gamma`. [#72](https://github.com/jiangyi15/tf-pwa/pull/72)
- Support `spins` in final states.
  [#71](https://github.com/jiangyi15/tf-pwa/pull/71)
- New model and new options for decay.
  [#66](https://github.com/jiangyi15/tf-pwa/pull/66)
- `config.get_particle_function` method for function of mass dependent only
  parts in amplitude. [#64](https://github.com/jiangyi15/tf-pwa/pull/64)

**Changed**

- More options in `config.plot_partial_wave`.
  [#73](https://github.com/jiangyi15/tf-pwa/pull/73)
- New algorithm to build decay chain in generating phsp.
  [#68](https://github.com/jiangyi15/tf-pwa/pull/68)

## [v0.1.6](https://github.com/jiangyi15/tf-pwa/tree/v0.1.6) (2021-10-28)

[Full Changelog](https://github.com/jiangyi15/tf-pwa/compare/v0.1.5...v0.1.6)

**<font color=red>Numeric</font>**

- Support parity transfomation, for charge conjugation process. The old method
  is still aviable for `cp_trans: False`
  [#53](https://github.com/jiangyi15/tf-pwa/pull/53)

- Update default options: `center_mass: False`, `r_boost: True`,
  `random_z: True`. [#60](https://github.com/jiangyi15/tf-pwa/pull/60)

**Added**

- Support Custom DecayChain. [#59](https://github.com/jiangyi15/tf-pwa/pull/59)
- Support to export `Saved Model` format of tensorflow.
  [#58](https://github.com/jiangyi15/tf-pwa/pull/58)
- Error propagation with automatic differentiation.
  [#56](https://github.com/jiangyi15/tf-pwa/pull/56)
  [#57](https://github.com/jiangyi15/tf-pwa/pull/57)
- New Decay model `gls-cpv` for CP violation.
  [#55](https://github.com/jiangyi15/tf-pwa/pull/55)
- Support Identical particles.
  [#54](https://github.com/jiangyi15/tf-pwa/pull/54)

**Changed**

- Force hessian matrix positive defined by adding some value for eigen value.
  [#52](https://github.com/jiangyi15/tf-pwa/pull/52)

## [v0.1.5](https://github.com/jiangyi15/tf-pwa/tree/v0.1.5) (2021-07-25)

[Full Changelog](https://github.com/jiangyi15/tf-pwa/compare/v0.1.4...v0.1.5)

**<font color=red>Numeric</font>**

- Fix bugs of aligned angle of wrong order of boost and rotation.
  [#51](https://github.com/jiangyi15/tf-pwa/pull/51)

**Added**

- Polarization and density matrix.
  [#45](https://github.com/jiangyi15/tf-pwa/pull/45)
- Plot for cfit. [#48](https://github.com/jiangyi15/tf-pwa/pull/48)
- Some possible optimize options.
  [#46](https://github.com/jiangyi15/tf-pwa/pull/46)
  [#47](https://github.com/jiangyi15/tf-pwa/pull/47)
  [#49](https://github.com/jiangyi15/tf-pwa/pull/49)
- New method of generate toy.
  [#50](https://github.com/jiangyi15/tf-pwa/pull/50)

## [v0.1.4](https://github.com/jiangyi15/tf-pwa/tree/v0.1.4) (2021-04-28)

[Full Changelog](https://github.com/jiangyi15/tf-pwa/compare/v0.1.3...v0.1.4)

**<font color=red>Numeric</font>**

- fix bugs of D function [#43](https://github.com/jiangyi15/tf-pwa/pull/43)

## [v0.1.3](https://github.com/jiangyi15/tf-pwa/tree/v0.1.3) (2021-04-28)

[Full Changelog](https://github.com/jiangyi15/tf-pwa/compare/v0.1.2...v0.1.3)

**Added**

- Model: BWR_LS [#33](https://github.com/jiangyi15/tf-pwa/pull/33)
- More options in fit.py
  [#35](https://github.com/jiangyi15/tf-pwa/pull/35)[#38](https://github.com/jiangyi15/tf-pwa/pull/38)
- C parity selection, `c_break: False`
  [#36](https://github.com/jiangyi15/tf-pwa/pull/36)

**Numeric**

- Use infered constant mass for calculating |p0| instead of mass in data.
  [#32](https://github.com/jiangyi15/tf-pwa/pull/32)

## [v0.1.2](https://github.com/jiangyi15/tf-pwa/tree/v0.1.2) (2021-03-20)

[Full Changelog](https://github.com/jiangyi15/tf-pwa/compare/v0.1.1...v0.1.2)

**Added**

- Resolution [#23](https://github.com/jiangyi15/tf-pwa/pull/23)
- Kmatrix in single channel and multiple pole.
  [#25](https://github.com/jiangyi15/tf-pwa/pull/25)
- Add `decay_params` and `production_params` in particle config.
  [#28](https://github.com/jiangyi15/tf-pwa/pull/28)
- Add `CalAngleData` class for all data after cal_angle.
  [#29](https://github.com/jiangyi15/tf-pwa/pull/29)
- Add more general method of phasespace generating, and interface in
  ConfigLoader. [#30](https://github.com/jiangyi15/tf-pwa/pull/30)

**Changed**

- Split `amp` and `config_loader` module.
  [#24](https://github.com/jiangyi15/tf-pwa/pull/24)
  [#26](https://github.com/jiangyi15/tf-pwa/pull/26)
- Change the VarsManager with bound.
  [#26](https://github.com/jiangyi15/tf-pwa/pull/26)
- Use angle_amp in `cached_amp: True` instead of amp/m_dep
  [#27](https://github.com/jiangyi15/tf-pwa/pull/27)
- Fix wrong LASS model.
  [#52d3eda](https://github.com/jiangyi15/tf-pwa/commit/52d3eda7b1cda7315764666aef329245735fef78)

## [v0.1.1](https://github.com/jiangyi15/tf-pwa/tree/v0.1.1) (2021-02-12)

[Full Changelog](https://github.com/jiangyi15/tf-pwa/compare/v0.1.0...v0.1.1)

**Added**

- Histogram module [#19](https://github.com/jiangyi15/tf-pwa/pull/19)
  [#22](https://github.com/jiangyi15/tf-pwa/pull/22)
- Plot decay struct without graphviz.
  [#20](https://github.com/jiangyi15/tf-pwa/pull/20)
- Model with expr for transpose model.
  [#21](https://github.com/jiangyi15/tf-pwa/pull/21)

**Changed**

- Save fit fractions as csv file in fit.py
- Allow `set_config("polar", False)` for parameters in cartesian coordinates.

**Numeric**

- Opposite angle
  ![0](https://latex.codecogs.com/gif.latex?\phi&space;+&space;\pi&space;\rightarrow&space;\phi&space;-&space;\pi)
  [#59d2fad](https://github.com/jiangyi15/tf-pwa/commit/59d2fad750ef3ccc9f3a5aed8b4ae6b8560d527f)

## [v0.1.0](https://github.com/jiangyi15/tf-pwa/tree/v0.1.0) (2021-01-15)

Backup vesion before some changes.

[Full Changelog](https://github.com/jiangyi15/tf-pwa/compare/v0.0.3...v0.1.0)

**Added**

- Interpolating resonances
  [#a261239](https://github.com/jiangyi15/tf-pwa/tree/a261239c1c5f3a86dab0630e851c37e972a17a58)
- Data mode
  [#cca472a](https://github.com/jiangyi15/tf-pwa/commit/cca472a3a05223256091fa7ec3f70ccef41e4d27)
- CFit [#13](https://github.com/jiangyi15/tf-pwa/pull/13)
- Some constrains configureation
- Some examples of deacy modules.

**Changed**

- Code style in [black](https://github.com/psf/black)
- Plot style [#9](https://github.com/jiangyi15/tf-pwa/pull/9)

## [v0.0.3](https://github.com/jiangyi15/tf-pwa/tree/v0.0.3) (2020-07-25)

[Full Changelog](https://github.com/jiangyi15/tf-pwa/compare/v0.0.2...v0.0.3)

Configuration file system

## [v0.0.2](https://github.com/jiangyi15/tf-pwa/tree/v0.0.2) (2020-03-16)

Automatic amplitude system.
