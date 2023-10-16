import yaml

from tf_pwa.config_loader import ConfigLoader
from tf_pwa.tests.common import write_temp_file

config_str = """

data:
    data: "data.dat"
    phsp: "PHSP.dat"

decay:
    ee:
    - [R1, b]
    - - R2
      - c
    - [R3, d, model: default]
    R1: [c, d, l_list: [1]]
    R2:
      - b
      - d
    R3: [b, c]

particle:
    $top:
        ee:
            J: 1
            P: -1
            spins: [-1, 1]
    $finals:
        b: {{J: 1, P: -1}}
        c: {{J: 1, P: -1, mass: 1.8}}
        d: {{J: 0, P: -1, mass: 0.1}}
    R1: [R1_a, R1_b]
    R2: {{J: 1, P: -1, mass: 2.3, width: 0.03}}
    R3: {{J: 1, P: -1, mass: 2.3, width: 0.03}}
    $include: "{file_name}"

"""

resonancs_str = """

R1_a: {J: 1, P: -1, mass: 2.3, width: 0.03}
R1_b: {J: 1, P: -1, mass: 2.3, width: 0.03}

"""


def test_load():
    with write_temp_file(resonancs_str) as f:
        cs = config_str.format(file_name=f)
        print(cs)
        with write_temp_file(cs) as g:
            config = ConfigLoader(g)
            with open(g) as f:
                data = yaml.full_load(f)
            config2 = ConfigLoader(data)
    config.get_amplitude()
    config2.get_amplitude()


def test_constrains():
    with write_temp_file(resonancs_str) as f:
        cs = config_str.format(file_name=f)
        print(cs)
        with write_temp_file(cs) as g:
            config = ConfigLoader(g)

    amp = config.get_amplitude()
    config.add_free_var_constraints(amp)


def test_decay_cut():
    def null_cut(decay_chain):
        return True, ""

    from tf_pwa.config_loader.decay_config import DecayConfig

    DecayConfig.decay_chain_cut_list["null"] = null_cut

    res_str = """
R1_a: {J: 1, P: -1, mass: 0.2, width: 0.03}
R1_b: {J: 0, P: +1, mass: 2.3, width: 0.03}
    """

    with write_temp_file(res_str) as f:
        cs = config_str.format(file_name=f)
        cs.replace(
            "data:",
            """data:
    decay_chain_cut: ["null", "mass_cut"]""",
        )
        print(cs)
        with write_temp_file(cs) as g:
            config = ConfigLoader(g)

    amp = config.get_amplitude()
