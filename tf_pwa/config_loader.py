import yaml
from tf_pwa.amp import get_particle, get_decay
from tf_pwa.particle import split_particle_type
import re

class ConfigLoader(object):
    """class for loading config.yml"""
    def __init__(self, file_name):
        self.config = self.load_config(file_name)
        self.particle_key_map = {
            "Par": "P",
            "m0": "mass",
            "g0": "width",
            "J" : "J",
            "P" : "P",
            "spins" :"spins",
            "bw": "model",
            "model": "model"
        }
        self.decay_key_map = {
            "model": "model"
        }
        self.dec = self.decay_item(self.config["decay"])
        self.particle_map, self.particle_property, self.top, self.finals  = self.particle_item(self.config["particle"])
        self.full_decay = self.get_decay_struct(self.dec, self.particle_map, self.particle_property, self.top, self.finals)
        self.decay_struct = self.get_decay_struct(self.dec)

    @staticmethod
    def load_config(file_name):
        with open(file_name) as f:
            ret = yaml.safe_load(f)
        return ret
    
    
    def get_decay(self, full=True):
        if full:
            return self.full_decay
        else:
            return self.decay_struct

    @staticmethod
    def _is_params(s):
        return "=" in s
    
    @staticmethod
    def _get_params(s):
        s = re.sub(r'^\s*', '',s) # skip blanks
        if s.startswith('"'):
            s = s[1:]
            km = re.match(r'^([^\"]+)\"\s*=', s)
            k = km.group(1)
            s = re.sub(r'^\s*', '', s[km.end():])
        else:
            km = re.match(r'^(.+)\s*=', s)
            k = km.group(1)
            k = re.sub(r'\s*$', '', k)
            s = re.sub(r'^\s*', '', s[km.end():])
        if s.startswith('"'):
            is_str = True
            s = s[1:]
            km = re.match(r'^([^\"]+)\"\s*$', s)
            v = km.group(1)
        else:
            is_str = False
            km = re.match(r'^(.+)\s*$', s)
            v = km.group(1)
            v = re.sub(r'\s*$', '', v)
        if not is_str:
            try: 
                v = int(v)
            except ValueError:
                try: 
                    v = float(v)
                except ValueError:
                    pass
        return k, v

    @staticmethod
    def _list2decay(core, outs):
        parts = []
        params = {}
        for j in outs:
            if ConfigLoader._is_params(j):
                k, v = ConfigLoader._get_params(j)
                params[k] = v
            else:
                parts.append(j)
        dec = {"core": core, "outs": parts, "params": params}
        return dec
    
    @staticmethod
    def decay_item(decay_dict):
        decs = []
        for core, outs in decay_dict.items():
            is_list = [isinstance(i, list) for i in outs]
            if all(is_list):
                for i in outs:
                    dec = ConfigLoader._list2decay(core, i)
                    decs.append(dec)
            else:
                dec = ConfigLoader._list2decay(core, outs)
                decs.append(dec)
        return decs

    @staticmethod
    def _do_include_dict(d, o):
        s = ConfigLoader.load_config(o)
        for i in s:
            if i not in d:
                d[i] = s[i]

    @staticmethod
    def particle_item_list(particle_list):
        particle_map = {}
        particle_property = {}
        for particle, candidate in particle_list.items():
            if isinstance(candidate, list):  # particle map
                for i in candidate:
                    if isinstance(i, str):                    
                        particle_map[particle] = particle_map.get(particle, []) + [i]
                    elif isinstance(i, dict):
                        map_i, pro_i = ConfigLoader.particle_item(i)
                        for k, v in map_i.items():
                            particle_map[k] = particle_map.get(k, []) + v
                        particle_property.update(pro_i)
                    else:
                        raise ValueError("vaule of particle map {} is {}".format(i, type(i)))
            elif isinstance(candidate, dict):
                particle_property[particle] = candidate
            else:
                raise ValueError("vaule of particle {} is {}".format(particle, type(candidate)))
        return particle_map, particle_property
                        
    @staticmethod
    def particle_item(particle_list):
        top = particle_list.pop("$top", None)
        finals = particle_list.pop("$finals", None)
        includes = particle_list.pop("$include", None)
        if includes:
            if isinstance(includes, list):
                for i in includes:
                    ConfigLoader._do_include_dict(particle_list, i)
            elif isinstance(includes, str):
                ConfigLoader._do_include_dict(particle_list, includes)
            else:
                raise ValueError("$include must be string or list of string not {}"
                                .format(type(includes)))
        particle_map, particle_property = ConfigLoader.particle_item_list(particle_list)
        
        if isinstance(top, dict):
            particle_property.update(top)
        if isinstance(finals, dict):
            particle_property.update(finals)
        return particle_map, particle_property, top, finals 

    def rename_params(self, params, is_particle=True):
        ret = {}
        if is_particle:
            key_map = self.particle_key_map
        else:
            key_map = self.decay_key_map
        for k, v in params.items():
            if k in key_map:
                ret[key_map[k]] = v
        return ret

    def get_decay_struct(self, decay, particle_map=None, particle_params=None, top=None, finals=None):
        """  get decay structure for decay dict"""
        particle_map = particle_map if particle_map is not None else {}
        particle_params = particle_params if particle_params is not None else {}
        
        particle_set = {}
        
        def add_particle(name):
            if name in particle_set:
                return particle_set[name]
            params = particle_params.get(name, {})
            params = self.rename_params(params)
            part = get_particle(name, **params)
            particle_set[name] = part
            return part
        
        def wrap_particle(name):
            name_list = particle_map.get(name, [name])
            return [add_particle(i) for i in name_list]
        
        def all_combine(outs):
            if len(outs) < 1:
                yield []
            else:
                for i in outs[0]:
                    for j in all_combine(outs[1:]):
                        yield [i] + j

        decs = [] 
        for dec in decay:
            core = wrap_particle(dec["core"])
            outs = [wrap_particle(j) for j in dec["outs"]]
            for i in core:
                for j in all_combine(outs):
                    dec_i = get_decay(i, j, **dec["params"])
                    decs.append(dec_i)

        if top is None or finals is None:
            top_tmp, res, finals_tmp = split_particle_type(decs)
        if top is None:
            top_tmp = list(top_tmp)
            assert len(top_tmp) == 1, "not only one top particle"
            top = list(top_tmp)[0]
        else: 
            if isinstance(top, str):
                top = particle_set(top)
            elif isinstance(top, dict):
                keys = list(top.keys())
                assert len(keys) == 1
                top = particle_set[keys.pop()]
            else:
                return particle_set[str(top)]
        if finals is None:
            finals = list(finals_tmp)
        elif isinstance(finals, (list, dict)):
            finals = [particle_set[i] for i in finals]
        else:
            raise TypeError("{}: {}".foramt(finals, type(finals)))
        
        dec_chain = top.chain_decay()
        return dec_chain
    
    def get_data_index(self, sub, name):
        dec = self.decay_struct
        if sub == "mass":
            p = get_particle(name)
            return ("particle", p, "m")
        if sub == "angle":
            de, de_i = None, None
            name_i = name.split("/")
            if len(name_i) > 1:
                _id = int(name_i[-1])
            else:
                _id = 0
            p = get_particle(name_i[0])
            for idx, i in enumerate(dec):
                for j in i:
                    if j.core == p:
                        de = j.core.decay[_id]
                    if j == de:
                        de_i = idx
            if de is None or de_i is None:
                raise ValueError("not found {}".format(name))
            return ("decay", de_i, de, de.outs[0], "ang")
        raise ValueError("unknown sub {}".format(sub))



