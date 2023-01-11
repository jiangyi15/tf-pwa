import contextlib
import itertools
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml

from tf_pwa.config_loader import ConfigLoader
from tf_pwa.config_loader.plot import (
    LineStyleSet,
    _get_cfit_bg,
    _get_cfit_eff_phsp,
    create_chain_property,
)
from tf_pwa.data import batch_call, data_index, data_shape
from tf_pwa.histogram import Hist1D, WeightedData

logger = logging.getLogger(__file__)


class PlotData:
    def __init__(
        self, dataset, weight=None, partial_weight=None, use_weighted=False
    ):
        self.dataset = dataset
        self.n_data = data_shape(dataset)
        self.weight = np.ones(self.n_data) if weight is None else weight
        self.partial_weight = {} if partial_weight is None else partial_weight
        self.scale = 1.0
        self.use_weighted = use_weighted

    def total_size(self):
        return np.sum(self.weight) * self.scale

    def __mul__(self, frac):
        if isinstance(frac, (float, int)):
            from copy import copy

            ret = copy(self)
            ret.scale = self.scale * frac
            return ret
        raise NotImplementedError()

    def get_histogram(self, var, partial=None, **kwargs):
        value = var(self.dataset)
        if partial is not None:
            w = self.partial_weight[partial]
        else:
            w = self.weight
        if self.use_weighted:
            return WeightedData(value, weights=w * self.scale, **kwargs)
        else:
            return Hist1D.histogram(value, weights=w * self.scale, **kwargs)


class ReadData:
    def __init__(self, var, trans=None):
        self.var = var
        self.trans = (lambda x: x) if trans is None else trans

    def __call__(self, data):
        value = data_index(data, self.var)
        value = self.trans(value)
        return value

    def __repr__(self):
        return str(self.var)


class PlotDataGroup:
    def __init__(self, datasets):
        self.datasets = datasets
        self.scale = 1.0
        self.n_data = sum(i.n_data for i in self.datasets)

    def total_size(self):
        return sum([np.sum(i.weight) for i in self.datasets]) * self.scale

    def get_histogram(self, var, partial=None, **kwargs):
        hist = [i.get_histogram(var, partial, **kwargs) for i in self.datasets]
        tmp = hist[0]
        for i in hist[1:]:
            tmp = tmp + i
        return tmp


class Frame:
    def __init__(
        self,
        var,
        x_range=None,
        nbins=None,
        name=None,
        display=None,
        trans=None,
        **extra
    ):
        self.var = var
        self.x_range = x_range
        self.nbins = nbins
        self.trans = trans
        self.var_trans = ReadData(var, trans)
        self.name = str(var) if name is None else name
        self.display = self.name if display is None else display
        self.extra = extra
        self.no_below = True

    def get_histogram(self, data, partial=None, bin_scale=1):
        if self.nbins is None:
            n_bins = max(5, min(data.n_data // 25, 200))
            hist = (
                data.get_histogram(
                    self.var_trans,
                    partial=partial,
                    range=self.x_range,
                    bins=n_bins,
                )
                * bin_scale
            )
        else:
            hist = (
                data.get_histogram(
                    self.var_trans,
                    partial=partial,
                    range=self.x_range,
                    bins=self.nbins * bin_scale,
                )
                * bin_scale
            )
        if self.x_range is None:
            self.x_range = hist.binning[0], hist.binning[-1]
        if self.nbins is None:
            self.nbins = (len(hist.binning) - 1) // bin_scale
        return hist

    def set_axis(self, axis, **config):
        if axis is plt:
            axis = plt.gca()
        config = {**self.extra, **config}
        delta = (self.x_range[1] - self.x_range[0]) / self.nbins
        if config.get("units", "") == "":
            axis.set_xlabel(self.display)
            axis.set_ylabel("Events/{:.3g}".format(delta))
        else:
            axis.set_xlabel(self.display + config.get("units", ""))
            axis.set_ylabel(
                "Events/({:.3g}  {})".format(delta, config.get("units", ""))
            )
        lower = None
        if self.no_below:
            lower = 0
        axis.set_ylim((lower, config.get("upper_ylim", None)))
        if "yscale" in config:
            axis.set_yscale(config["yscale"])
        if "has_legend" in config:
            if config["has_legend"]:
                if "legend.fontsize" in config:
                    axis.legend(fontsize=config["legend.fontsize"])
                else:
                    axis.legend()


@ConfigLoader.register_function()
def get_all_frame(self):
    ret = {}
    for conf in self.plot_params.get_params():
        name = conf.get("name")
        display = conf.get("display", name)
        upper_ylim = conf.get("upper_ylim", None)
        idx = conf.get("idx")
        trans = conf.get("trans", lambda x: x)
        has_legend = conf.get("legend", False)
        xrange = conf.get("range", None)
        bins = conf.get("bins", None)
        units = conf.get("units", "")
        yscale = conf.get("yscale", "linear")
        ret[name] = Frame(
            idx,
            trans=trans,
            name=name,
            display=display,
            x_range=xrange,
            nbins=bins,
            units=units,
            yscale=yscale,
            has_legend=has_legend,
            upper_ylim=upper_ylim,
        )
    return ret


class PlotAllData:
    def __init__(self, amp, data, phsp, bg=None, res=None, use_weighted=False):
        self.amp = amp
        self.data = data
        self.phsp = phsp
        self.bg = bg
        self.res = res
        self.datasets = {
            "data": PlotData(
                data, weight=data.get_weight(), use_weighted=use_weighted
            ),
            "fitted": PlotData(
                phsp,
                weight=phsp.get_weight() * batch_call(amp, phsp),
                use_weighted=use_weighted,
            ),
        }
        if self.bg is not None:
            self.datasets["bg"] = PlotData(
                bg, weight=-bg.get_weight(), use_weighted=use_weighted
            )
        if res is None:
            weights = amp.partial_weight(phsp)
            new_res = [str(i) for i in amp.decay_group]
        else:
            weights = []
            used_res = amp.used_res
            new_res = []
            for i in res:
                if not isinstance(i, list):
                    i = [i]
                new_res.append(tuple(i))
                amp.set_used_res(i)
                weights.append(amp(phsp))
            # print(weights, amp.decay_group.chains_idx)
            amp.set_used_res(used_res)
        self.datasets["fitted"].partial_weight = dict(zip(new_res, weights))

        if self.bg is None:
            n_sig = self.datasets["data"].total_size()
        else:
            n_sig = (
                self.datasets["data"].total_size()
                - self.datasets["bg"].total_size()
            )
        self.datasets["fitted"].scale = (
            n_sig / self.datasets["fitted"].total_size()
        )

    def get_all_histogram(self, var, bin_scale=3):
        ret = {}
        for k, v in self.datasets.items():
            ret[(k,)] = var.get_histogram(v)
            for k2 in v.partial_weight:
                ret[(k, k2)] = var.get_histogram(
                    v, partial=k2, bin_scale=bin_scale
                )
        return ret


def merge_hist(hists):
    hists = list(hists)
    ret = {k: v for k, v in hists[0].items()}
    for i in hists[1:]:
        for k in ret:
            ret[k] = ret[k] + i[k]
    return ret


class StyleSet:
    def __init__(self, file_name):
        self.file_name = file_name
        if file_name is not None:
            with open(file_name) as f:
                all_style = yaml.full_load(f)
            all_style = {} if all_style is None else all_style
            self.all_style = {}
            for k in all_style:
                id_ = k.pop("id")
                self.all_style[id_] = k
        else:
            self.all_style = {}
        self.prop_cycler = itertools.cycle(plt.rcParams["axes.prop_cycle"])

    def get(self, key, value=None):
        if key in self.all_style:
            return self.all_style[key]
        if value is not None:
            self.all_style[key] = value
            return self.all_style[key]
        self.all_style[key] = self.generate_new_style()
        return self.all_style[key]

    def set(self, key, value):
        self.all_style[key] = {**self.all_style.get(key, {}), **value}

    def generate_new_style(self):
        return next(self.prop_cycler)

    def save(self):
        with open(self.file_name, "w") as f:
            all_style = [{"id": k, **v} for k, v in self.all_style.items()]
            yaml.dump(all_style, f, indent=2)


class Plotter:
    def __init__(self, config, legend_file=None, res=None, use_weighted=False):
        self.config = config
        self.componets = config.get_all_components(
            res=res, use_weighted=use_weighted
        )
        self.res = res
        self.chain_property = create_chain_property(config, res=res)
        self.all_frame = config.get_all_frame()
        self.style = StyleSet(legend_file)
        self.plot_item = None
        self.extra_plot_item = []
        self.hidden_plot_item = []
        self.small_bg = True
        self.smooth = False
        self.draw_item = {}
        self.default_plot_style = {
            (("data",),): {
                "label": "data",
                "type": "error",
                "color": "black",
                "linestyle": "none",
                "marker": ".",
                "zorder": 100,
            },
            (("bg",),): {
                "label": "background",
                "type": "bar",
                "color": "grey",
                "alpha": 0.5,
            },
            (("fitted",), ("bg",)): {
                "label": "total fit",
                "type": "hist",
                "color": "black",
                "linestyle": "-",
                "marker": "",
                "zorder": 99,
            },
            (("fitted",),): {
                "label": "total fit",
                "type": "hist",
                "color": "black",
                "linestyle": "-",
                "marker": "",
                "zorder": 99,
            },
        }

    def get_label(self, key):
        if len(key) == 1:
            head = key[0]
            if len(head) == 2:
                if head[0] == "fitted":
                    dic = head[1]

    def get_plot_style(self, example_hist):
        for k, v in example_hist.items():
            if k == ("data",):
                self.style.get((k,), self.default_plot_style[(k,)])
            elif k == ("fitted",):
                if ("bg",) in example_hist:
                    key = (("fitted",), ("bg",))
                    self.style.get(key, self.default_plot_style[key])
                    self.style.get((("fitted",),), {})
                else:
                    self.style.get(
                        (("fitted",),), self.default_plot_style[(k,)]
                    )
            elif k == ("bg",):
                self.style.get((k,), self.default_plot_style[(k,)])
            elif k[0] == "fitted":
                style = {}
                if self.smooth:
                    style["type"] = "kde"
                extra_style = self.get_res_style(k[1])
                self.style.get((k,), {**style, **extra_style})
            else:
                pass
                # if self.smooth:
                #    self.style.get((k,), {"type": "kde"})
                # else:
                #    self.style.get((k,))
        return self.style

    def get_res_style(self, key):
        if isinstance(key, str):
            for i, j in enumerate(self.config.get_decay()):
                if str(j) == key:
                    style = self.chain_property[i]
                    return {"name": style[1], "label": style[2]}
        elif isinstance(key, tuple):
            for i, name in enumerate(self.res):
                if not isinstance(name, list):
                    name = [name]
                if tuple(name) == key:
                    style = self.chain_property[i]
                    return {"name": style[1], "label": style[2]}
        return {}

    def set_plot_item(self, example_hist):
        self.plot_item = []
        for k, v in example_hist.items():
            self.plot_item.append((k,))
        if ("bg",) in example_hist:
            if (("fitted",), ("bg",)) not in self.plot_item:
                self.plot_item.append((("fitted",), ("bg",)))
            if self.small_bg:
                self.plot_item.remove((("fitted",),))

    def plot_frame(self, name, idx=None, ax=plt, bin_scale=3):
        frame = self.all_frame.get(name, None)
        if frame is None:
            raise IndexError("no such frame")
        return self.plot_var(frame, idx, ax=ax, bin_scale=bin_scale)

    def get_all_hist(self, frame, idx=None, bin_scale=3):
        if idx is None:
            hists = merge_hist(
                i.get_all_histogram(frame, bin_scale=bin_scale)
                for i in self.componets
            )
        else:
            hists = self.componets[idx].get_all_histogram(
                frame, bin_scale=bin_scale
            )
        return hists

    def plot_var(self, frame, idx=None, ax=plt, bin_scale=3):
        hists = self.get_all_hist(frame, idx=idx, bin_scale=bin_scale)
        plot_style = self.get_plot_style(hists)
        if self.plot_item is None:
            self.set_plot_item(hists)
        no_below = True
        for i in self.extra_plot_item + self.plot_item:
            if i in self.hidden_plot_item:
                continue
            hist = hists[i[0]]
            for j in i[1:]:
                hist = hist + hists[j]
            style = self.style.get(i)
            style = {k: v for k, v in style.items() if k not in ["name", "id"]}
            a = hist.draw(ax, **style)
            no_below = no_below and np.all(hist.count > 0)
            self.draw_item[i] = a
        if isinstance(frame, Frame):
            frame.no_below = no_below
            frame.set_axis(ax)
        return ax

    def forzen_style(self):
        assert len(self.draw_item) > 0
        import matplotlib as mpl

        for key in self.draw_item:
            style = self.style.get(key)
            plot_type = style.get("type", "hist")
            if plot_type == "hist" in ["line", "error", "kde", "hist"]:
                art = self.draw_item[key][0]
                tmp = {
                    "color": art.get_color(),
                    "linestyle": art.get_linestyle(),
                    "marker": art.get_marker(),
                }
                self.style.set(key, tmp)
            elif plot_type == "bar":
                art = self.draw_item[key][0]
                tmp = {"facecolor": mpl.colors.to_hex(art.get_facecolor())}
                self.style.set(key, tmp)
            else:
                pass

    @contextlib.contextmanager
    def old_style(self, extra_config=None, color_first=True):
        import matplotlib as mpl
        from cycler import cycler

        extra_config = {} if extra_config is None else extra_config
        colors = [
            "red",
            "orange",
            "purple",
            "springgreen",
            "y",
            "green",
            "blue",
            "c",
        ]
        linestyles = ["-", "--", "-.", ":"]
        marker = [",", ".", "^"]
        if color_first:
            style = (
                cycler(marker=marker)
                * cycler(linestyle=linestyles)
                * cycler(color=colors)
            )
        else:
            style = (
                cycler(marker=marker)
                * cycler(color=colors)
                * cycler(linestyle=linestyles)
            )
        config = {
            "axes.prop_cycle": style,
            "legend.frameon": False,
            "legend.borderpad": 0.0,
            "legend.labelspacing": 0.1,
        }
        total_config = {**config, **extra_config}
        with mpl.rc_context(total_config):
            yield total_config

    def plot_frame_with_pull(
        self, name, idx=None, bin_scale=3, pull_config=None
    ):
        frame = self.all_frame.get(name, None)
        if frame is None:
            raise IndexError("no such frame")
        hists = self.get_all_hist(frame, idx=idx, bin_scale=bin_scale)
        # import matplotlib.gridspec as gridspec
        # gs = gridspec.GridSpec(4,1)
        ax = plt.subplot2grid(
            (4, 1), (0, 0), rowspan=3
        )  # plt.subplot(gs[:3,0])
        ax2 = plt.subplot2grid(
            (4, 1), (3, 0), rowspan=1
        )  # plt.subplot(gs[3:,0])
        self.plot_var(frame, idx, ax=ax, bin_scale=bin_scale)
        if ("bg",) in hists:
            total_fit = hists[("bg",)] + hists[("fitted",)]
        else:
            total_fit = hists[("fitted",)]
        pull = hists[("data",)] - total_fit
        pull_config = {} if pull_config is None else pull_config
        pull.draw_pull(ax2, **pull_config)
        frame.set_axis(ax2, has_legend=False)
        ax2.axhline(y=0, color="r", linewidth=0.5)
        ax2.axhline(
            y=3,
            color="r",
            linestyle="--",
            linewidth=0.5,
        )
        ax2.axhline(
            y=-3,
            color="r",
            linestyle="--",
            linewidth=0.5,
        )
        ax2.set_ylim((-5, 5))
        ax2.set_ylabel("pull")
        ax2.set_yscale("linear")
        ax.set_xticks([])
        ax.set_xlabel("")
        return ax, ax2

    def save_all_frame(
        self,
        prefix="figure/",
        format="png",
        idx=None,
        plot_pull=False,
        pull_config=None,
    ):
        path = os.path.dirname(prefix)
        os.makedirs(path, exist_ok=True)
        for name in self.all_frame:
            plt.clf()
            if plot_pull:
                ax, _ = self.plot_frame_with_pull(
                    name, idx=idx, pull_config=pull_config
                )
            else:
                ax = self.plot_frame(name)
            logger.info("save figure: ", prefix + name + "." + format)
            plt.savefig(prefix + name + "." + format)

    def add_ref_amp(self, ref_amp, name="reference fit"):
        for i in self.componets:
            base_data = i.datasets["fitted"]
            phsp = base_data.dataset
            i.datasets[name] = PlotData(
                phsp, ref_amp(phsp) * phsp.get_weight()
            )
            i.datasets[name].scale = (
                base_data.total_size() / i.datasets[name].total_size()
            )
        if "bg" in self.componets[0].datasets:
            key = ((name,), ("bg",))
            self.hidden_plot_item.append(((name,),))
        else:
            key = ((name,),)
        self.extra_plot_item.append(key)
        self.style.get(key, {"label": name, "linewidth": 2, "zorder": -1})


@ConfigLoader.register_function()
def get_plotter(self, legend_file=None, res=None, use_weighted=False):
    return Plotter(self, legend_file, res=res, use_weighted=use_weighted)


@ConfigLoader.register_function()
def get_all_components(
    self, data=None, phsp=None, bg=None, res=None, use_weighted=False
):
    amp = self.get_amplitude()
    data = self.get_data("data") if data is None else data
    phsp = self.get_data("phsp") if phsp is None else data
    bg = self.get_data("bg") if bg is None else bg
    if bg is None:
        if self.config["data"].get("model", "auto") == "cfit":
            bg = _get_cfit_bg(self, data, phsp)
        else:
            bg = [None] * len(data)
    if self.config["data"].get("model", "auto") == "cfit":
        phsp = _get_cfit_eff_phsp(self, phsp)
    return [
        PlotAllData(
            amp, data=a, phsp=b, bg=c, res=res, use_weighted=use_weighted
        )
        for a, b, c in zip(data, phsp, bg)
    ]
