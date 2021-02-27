import numpy as np

from tf_pwa.histogram import *


def test_hist1d():
    data = np.linspace(0, 1, 500)
    weight = np.cos((data - 0.5) * np.pi)
    hist = Hist1D.histogram(data, weights=weight, bins=30)
    plt.clf()
    ax = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
    hist.draw(ax)
    hist.draw_line()
    hist.draw_line(kind="quadratic")
    hist.draw_kde(ax, kind="gauss", color="blue")
    hist.draw_kde(ax, kind="cauchy", color="red")
    (0.1 * hist + hist * 0.1).draw_bar(ax)
    hist.draw_error(ax)
    hist2 = Hist1D.histogram(
        data,
        weights=weight
        + np.random.random(
            500,
        )
        * 0.1
        - 0.05,
        bins=30,
    )
    ax2 = plt.subplot2grid((4, 1), (3, 0), rowspan=1)
    (hist2 - hist).draw_pull(ax2)
    plt.savefig("hist1d_test1.png")


def test_weighteddata():
    data = np.linspace(0, 1, 500)
    weight = np.cos((data - 0.5) * np.pi)
    hist = WeightedData(data, weights=weight, bins=30)
    plt.clf()
    hist.draw()
    hist.draw_kde(kind="gauss", color="blue")
    hist.draw_kde(kind="uniform", color="green")
    hist.draw_kde(kind="epanechnikov", color="red")
    hist.draw_kde(kind=gauss, color="pink")
    hist2 = hist * 0.1 + hist
    plt.savefig("hist1d_test2.png")


def test_weight():
    data = np.linspace(0, 1, 500)
    weight = np.cos((data - 0.5) * np.pi)
    hist1 = Hist1D.histogram(data, weights=weight, bins=30)
    hist2 = Hist1D.histogram(data, bins=30)
    a, b = np.histogram(data, weights=weight)
    hist3 = Hist1D(b, a)
    hist4 = WeightedData(data, bins=30)
    assert hist1.get_count() == hist3.get_count()
    assert hist2.get_count() == hist4.get_count()
    assert hist1.scale_to(hist2) == 1 / np.mean(weight)
