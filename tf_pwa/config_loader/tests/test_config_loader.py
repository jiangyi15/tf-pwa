from tf_pwa.tests.common import write_temp_file
from tf_pwa.config_loader import ConfigLoader


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
    R1: [c, d, l_list: [0]]
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
        c: {{J: 1, P: -1}}
        d: {{J: 1, P: -1}}
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
    config.get_amplitude()
