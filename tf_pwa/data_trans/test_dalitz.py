from tf_pwa.data_trans.dalitz import *


def test_mass():
    m0 = 1.86
    m1 = 0.493
    m2 = 0.493
    m3 = 0.139

    def mass2(p):
        """mass2"""
        p0, p1, p2, p3 = tf.unstack(p, axis=-1)
        return p0 ** 2 - p1 ** 2 - p2 ** 2 - p3 ** 2

    def mass(p):
        """mass"""
        m2 = mass2(p)
        return tf.sqrt(m2)

    p1, p2, p3 = generate_p(
        np.array([1.3]) ** 2, np.array([1.23]) ** 2, m0, m1, m2, m3
    )

    assert np.allclose(mass(p1), m1)
    assert np.allclose(mass(p2), m2)
    assert np.allclose(mass(p3), m3)
    assert np.allclose(mass(p1 + p2 + p3), m0)
    assert np.allclose(mass(p1 + p2), 1.3)
    assert np.allclose(mass(p2 + p3), 1.23)


if __name__ == "__main__":
    test_mass()
