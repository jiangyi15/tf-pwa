
---------------------
Resonances Parameters
---------------------

This section is about how do the `Resonances.yml` work.

From `Resonances.yml` to the real model, there will be following steps.


1. loaded by config.yml, it is will be combined the definition in `config.yml` particle parts.

   For examples, `config.yml` have

   .. code::

      particle:
         $include: Resonances.yml
         Psi4040:
            float: mg

   then `Resonances.yml` item

   .. code::

      Psi4040:
         J: 1
         float: m

   will become `{"Psi4040": {"J": 1, "float": "mg"}}`

2. replace some alias, (`m0 -> mass`, `g0 -> width`, ...)

3. If it is used in decay chain, then create `Particle` object.

   The particle class is `cls = get_particle_model(item["model"])`,
   and the object is  `cls(**item)`.

   All parameters will be stored in `config.particle_property[name]`.



All available parameters can be divided into the following 3 sets.

Common Parameters
-----------------

Parameters defined in `~tf_pwa.particle.BaseParticle` are common parameters including spin, parity, mass and width.

+-----------+----------------+-------------------------------------------------------------------------------+
|  name     | default value  | comment                                                                       |
+===========+================+===============================================================================+
|   `J`     |     0          |  spin, int or half-integral                                                   |
+-----------+----------------+-------------------------------------------------------------------------------+
|   `P`     |     -1         |  P-parity, +1 or -1                                                           |
+-----------+----------------+-------------------------------------------------------------------------------+
|   `C`     |    None        |  C-Parity, +1 or -1                                                           |
+-----------+----------------+-------------------------------------------------------------------------------+
|  `mass`   |    None        |  mass, float, it is always required because of the calculation of :math:`q_0` |
+-----------+----------------+-------------------------------------------------------------------------------+
|  `width`  |    None        |  width, float                                                                 |
+-----------+----------------+-------------------------------------------------------------------------------+
|  `spins`  |    None        |  possible spin projections,`[-J, ..., J]`, list                               |
+-----------+----------------+-------------------------------------------------------------------------------+



Model Parameters
----------------

Parameters defined in the real model. :doc:`particle_model`


There are many parameters defined by the user, then those parameters will be passed to the model class,
such as the parameters for `__init__(self, **kwargs)` method.

For example, the default model (`BWR`, `~tf_pwa.particle.BaseParticle`) has the following parameters:

+---------------------+-----------------------------+----------------------------------------+
|  name               | default value               | comment                                |
+=====================+=============================+========================================+
| `running_width`     |     True                    |  if using running width, bool          |
+---------------------+-----------------------------+----------------------------------------+
|   `bw_l`            | None, auto determined       |  running width angular momentum, int   |
+---------------------+-----------------------------+----------------------------------------+


Other Parameters
----------------

There are also some other parameters which is used to control the program running.

For example, simple constraints, the following parameters are used by `~tf_pwa.config_loader.ConfigLoader` as constraints.

+----------------------------+----------------+-----------------------------------+
|  name                      | default value  | comment                           |
+============================+================+===================================+
| `mass_min`, `mass_max`     |     None       |  mass range                       |
+----------------------------+----------------+-----------------------------------+
| `width_min`, `width_max`   |     None       |  width range                      |
+----------------------------+----------------+-----------------------------------+
| `float`                    |    `[]`        |  float parameters list            |
+----------------------------+----------------+-----------------------------------+


Another example are parameters to build decay chain for particle `R`.

+------------------------+----------------+----------------------------------------------------------+
|  name                  | default value  | comment                                                  |
+========================+================+==========================================================+
| `decay_params`         |     `{}`       |  parameters pass to `Decay` which `R` decay              |
+------------------------+----------------+----------------------------------------------------------+
|  `production_params`   |     `{}`       |  parameters pass to `Decay` which produce `R`            |
+------------------------+----------------+----------------------------------------------------------+
|  `model`               |   `default`    |  Particle model for `R`                                  |
+------------------------+----------------+----------------------------------------------------------+

There are also other commonly used parameters.

+------------------------+----------------+----------------------------------------------------------+
|  name                  | default value  | comment                                                  |
+========================+================+==========================================================+
| `display`              |     `None`     |  control plot legend with latex string, string           |
+------------------------+----------------+----------------------------------------------------------+
