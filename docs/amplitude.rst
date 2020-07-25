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

