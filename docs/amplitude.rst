----------------
Amplitude
----------------


Helicity Formula
________________

Each Decay has Amplitude like

.. math::
    A^{A \rightarrow B+C}_{\lambda_{A},\lambda_{B},\lambda_{C}} = H_{\lambda_{B},\lambda_{C}}^{A \rightarrow B+C} D^{J_{A}\star}_{\lambda_{A},\lambda_{B}-\lambda_{C}}(\phi,\theta,0)

For a chain decay, amplitude can be combined as

.. math::
    A^{A \rightarrow R+B,R \rightarrow C+D}_{\lambda_{A},\lambda_{B},\lambda_{C},\lambda_{D}}
    = \sum_{\lambda_{R}}A^{A \rightarrow R+B}_{\lambda_{A},\lambda_{R},\lambda_{B}}
    \color{red}{R(m_{R})}\color{black} A^{R \rightarrow C+D} _{\lambda_{R},\lambda_{C},\lambda_{D}}

with angle aligned

.. math::
    {\hat{A}}^{A \rightarrow R+B,R \rightarrow C+D}_{\lambda_{A},\lambda_{B},\lambda_{C},\lambda_{D}}
    = \sum_{\lambda_{B}',\lambda_{C}',\lambda_{D}'}A^{A \rightarrow R+B,R \rightarrow C+D}_{\lambda_{A},\lambda_{B}',\lambda_{C}',\lambda_{D}'}
    D^{J_{B}\star}_{\lambda_{B}',\lambda_{B}}(\alpha_{B},\beta_{B},\gamma_{B})
    D^{J_{C}\star}_{\lambda_{C}',\lambda_{C}}(\alpha_{C},\beta_{C},\gamma_{C})
    D^{J_{D}\star}_{\lambda_{D}',\lambda_{D}}(\alpha_{D},\beta_{D},\gamma_{D})

the sum of resonances

.. math::
    A_{\lambda_{A},\lambda_{B},\lambda_{C},\lambda_{D}}^{total} = \sum_{R_{1}} {\hat{A}}^{A \rightarrow R_{1}+B,R_{1} \rightarrow C+D}_{\lambda_{A},\lambda_{B},\lambda_{C},\lambda_{D}}
    + \sum_{R_{2}} {\hat{A}}^{A \rightarrow R_{2}+C,R_{2} \rightarrow B+D}_{\lambda_{A},\lambda_{B},\lambda_{C},\lambda_{D}}
    + \sum_{R_{3}} {\hat{A}}^{A \rightarrow R_{3}+D,R_{3} \rightarrow B+C}_{\lambda_{A},\lambda_{B},\lambda_{C},\lambda_{D}}


then the differential cross-section

.. math::
    \frac{d\sigma}{d\Phi} = \frac{1}{N}\sum_{\lambda_{A}}\sum_{\lambda_{B},\lambda_{C},\lambda_{D}}|A_{\lambda_{A},\lambda_{B},\lambda_{C},\lambda_{D}}^{total}|^2

the LS coupling formula

.. math::
    H_{\lambda_{B},\lambda_{C}}^{A \rightarrow B+C} =
    \sum_{ls} g_{ls} \sqrt{\frac{2l+1}{2 J_{A}+1}} CG_{ls\rightarrow \lambda_{B},\lambda_{C}} \color{red}{f_{l}(q,q_0)}

Amplitude Combination Rules
---------------------------

For a decay process `A -> R B, R -> C D`, we can get different part of amplitude:

1. Particle:
    1. Initial state: :math:`1`

    2. Final state: :math:`D(\alpha, \beta, \gamma)`

    3. Propagator: :math:`R(m)`

2. Decay:
    Two body decay (`A -> R B`): :math:`H_{\lambda_R,\lambda_B} D_{\lambda_A, \lambda_R - \lambda_B} (\varphi, \theta,0)`

Now we can use combination rules to build amplitude for the whole process.

    Probability Density:
        :math:`P = |\tilde{A}|^2` (modular square)

        Decay Group:
            :math:`\tilde{A} = A_{R_1} + A_{R_2} + \cdots` (addition)

            Decay Chain:
                :math:`A_{R} = A_1 \times R \times A_2 \cdots` (multiplication)

                Decay:
                :math:`A_i = HD(\varphi, \theta, 0)`

                Particle:
                :math:`R(m)`

The indices part is quantum number, and it can be summed automatically.
