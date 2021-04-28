# Changelog

## [Unreleased](https://github.com/jiangyi15/tf-pwa/tree/HEAD)

[Full Changelog](https://github.com/jiangyi15/tf-pwa/compare/v0.1.2...HEAD)

## [v0.1.4](https://github.com/jiangyi15/tf-pwa/tree/v0.1.2) (2021-03-20)

[Full Changelog](https://github.com/jiangyi15/tf-pwa/compare/v0.1.3...v0.1.4)

**Numeric**

- fix bugs of D function [#43](https://github.com/jiangyi15/tf-pwa/pull/43)

## [v0.1.3](https://github.com/jiangyi15/tf-pwa/tree/v0.1.2) (2021-03-20)

[Full Changelog](https://github.com/jiangyi15/tf-pwa/compare/v0.1.2...v0.1.3)

**Added**

- Model: BWR_LS [#3](https://github.com/jiangyi15/tf-pwa/pull/33)
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
