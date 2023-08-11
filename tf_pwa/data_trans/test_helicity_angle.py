from tf_pwa.data_trans.helicity_angle import *
from tf_pwa.tests.test_full import gen_toy, toy_config


def test_gen_p():

    data = {
        k: np.array([v])
        for k, v in zip(
            [
                "mjpsip",
                "Z_cosTheta_Lb",
                "Z_cosTheta_Z",
                "Z_cosTheta_Jpsi",
                "Z_phiZ",
                "Z_phiJpsi",
                "Z_phiMu",
            ],
            [
                4.46178,
                0.808274,
                0.243913,
                0.0598463,
                1.88456,
                2.01922,
                1.35353,
            ],
        )
    }

    mLb = 5.6195
    mp = 0.938272081
    mJpsi = 3.0969
    mmu = 0.1056583745
    mK = 0.493677

    costheta = [
        data[i] for i in ["Z_cosTheta_Lb", "Z_cosTheta_Z", "Z_cosTheta_Jpsi"]
    ]
    phi = [data[i] for i in ["Z_phiZ", "Z_phiJpsi", "Z_phiMu"]]

    mjpsip = data["mjpsip"]
    a = generate_p([mLb, mjpsip, mJpsi, mmu], [mK, mp, mmu], costheta, phi)

    assert np.allclose(lv.M(a[0]), mK)
    assert np.allclose(lv.M(a[1]), mp)
    assert np.allclose(lv.M(a[2]), mmu)
    assert np.allclose(lv.M(a[3]), mmu)
    assert np.allclose(lv.M(a[0] + a[1] + a[2] + a[3]), mLb)
    assert np.allclose((a[0] + a[1] + a[2] + a[3])[..., 0], mLb)
    assert np.allclose(lv.M(a[1] + a[2] + a[3]), mjpsip)
    assert np.allclose(lv.M(a[2] + a[3]), mJpsi)


def test_gen_p_mass():
    from tf_pwa.amp import DecayChain, get_decay, get_particle

    mLb = 5.6195
    mp = 0.938272081
    mJpsi = 3.0969
    mmu = 0.1056583745
    mK = 0.493677

    p1 = get_particle("Lb", mass=5.6195)
    p2 = get_particle("jpsip", mass=4.46178)
    p3 = get_particle("K", mass=0.493677)
    p4 = get_particle("jspi", mass=3.0969)
    p5 = get_particle("p", mass=0.938272081)
    p6 = get_particle("mup", mass=0.1056583745)
    p7 = get_particle("mum", mass=0.1056583745)

    dec1 = get_decay(p1, [p2, p3])
    dec2 = get_decay(p2, [p4, p5])
    dec3 = get_decay(p4, [p6, p7])
    dec = DecayChain([dec1, dec2, dec3])
    ha = HelicityAngle1(dec)
    ha2 = HelicityAngle(dec)

    b = ha.generate_p_mass("jpsip", 4.46178)
    b2 = ha2.generate_p_mass("jpsip", 4.46178)
    for k in b:
        assert np.allclose(b[k], b2[k])

    def check_b(b):
        a = [b[i] for i in ha.par]
        assert np.allclose(lv.M(a[0]), mK)
        assert np.allclose(lv.M(a[1]), mp)
        assert np.allclose(lv.M(a[2]), mmu)
        assert np.allclose(lv.M(a[3]), mmu)
        assert np.allclose(lv.M(a[0] + a[1] + a[2] + a[3]), mLb)
        assert np.allclose((a[0] + a[1] + a[2] + a[3])[..., 0], mLb)
        assert np.allclose(lv.M(a[1] + a[2] + a[3]), 4.46178)
        assert np.allclose(lv.M(a[2] + a[3]), mJpsi)

    check_b(b)

    b = ha.generate_p_mass("jpsip", 4.46178, random=True)
    b2 = ha2.generate_p_mass("jpsip", 4.46178, random=True)

    check_b(b)
    check_b(b2)


def test_helicity_var(toy_config):
    decay_chain = toy_config.get_decay().get_decay_chain("R_BC")
    ha = HelicityAngle(decay_chain)
    toy = toy_config.generate_toy(100)

    var = ha.find_variable(toy)
    p_new = ha.build_data(*var)

    toy_new = toy_config.data.cal_angle(p_new)

    assert np.allclose(toy.get_mass("(B, C)"), toy_new.get_mass("(B, C)"))
    assert np.allclose(
        toy.get_angle("(B, C)", "B")["beta"],
        toy_new.get_angle("(B, C)", "B")["beta"],
    )
    assert np.allclose(
        toy.get_angle("(C, D)", "C")["alpha"],
        toy_new.get_angle("(C, D)", "C")["alpha"],
    )

    p_rec = ha.cal_angle(p_new)
    p_rec = ha.find_variable(p_rec)
    p_rec = ha.build_data(*p_rec)
    for k in p_new:
        np.allclose(p_rec[k], p_new[k])


if __name__ == "__main__":
    test_gen_p()
    test_gen_p_mass()
