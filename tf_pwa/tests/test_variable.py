from tf_pwa.variable import Variable


def test_variable():
    m = Variable("R_m", value=2.1)  # trainable is True by default
    g_ls = Variable("A2BR_H", shape=[3], cplx=True)
    fcr = Variable("R_total", cplx=True)
    m1 = Variable("R1_m", value=2.3)
    g_ls1 = Variable("A2BR1_H", shape=[3], cplx=True)
    fcr1 = Variable("R1_total", cplx=True)

    print(g_ls.value)
    print(g_ls())

    m.fixed(2.4)
    g_ls.sameas(g_ls1)
    fcr.r_shareto(fcr1)
