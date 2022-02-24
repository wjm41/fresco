# Work journal for FRESCOE

Fragment Ensemble Significant PharmaCOphore Extraction (FRESCOE) method for hit-finding from a fragment screen. FRESCOE uses unsupervised machine learning to model a distribution of pharmacophore combinations in 3D space, which is then used to conduct a virtual screen on the EnamineREAL library for discovery of moelcular hits. This method is unique in directly predicting hit compounds from a fragment screen without any assaying of binding activity required, and has demonstrated success computationally on a retrospective analysis of COVID Moonshot compounds as well as suggesting an experimentally validated novel scaffold for inhibiting the SARS-CoV-2 protease.

## Flowcharts

```mermaid
flowchart LR
subgraph "v2 (current)"
direction LR
subgraph EnamineREAL
A[(VirtualFlow)] --> B(Pharmacophore Descriptors)
end

subgraph Fragments
Mpro --> PharmD(Pharmacophore Distributions)
Mac-1 --> PharmD
DPP11 --> PharmD
PharmD --> KDExact(scikit-learn KDE)
KDExact --> Keep(Keep all pharmacophores!)
Keep --> KDE(scipy interpolation)
end

subgraph Virtual Screening
B --> E(Score Library)
KDE --> E
E --> F(Choose Top-50k)
F --> G(Filtering)
G -- "physchem (lead-like)\n plus constraint on \n donors + acceptors"--> H(Clustering)
G -- pains --> H
G -- "at most one chiral center" --> H
G -- removed duplicate tautomers --> H

end
end
```

```mermaid
flowchart LR
subgraph v1
direction LR
subgraph EnamineREAL
A[(VirtualFlow)] --> B(Pharmacophore Descriptors)
end

subgraph Fragments
Mpro --> PharmD(Pharmacophore Distributions)
Mac-1 --> PharmD
DPP11 --> PharmD
PharmD --> KDExact(scikit-learn KDE)
KDExact --> KDE(KS Testing for ''importance'')
end

subgraph Virtual Screening
B --> Score(Score library with ''important'' pharmacophores)
KDE --> Score
Score --> Choose(Choose Top-10k)
Choose --> G(Filtering)
G -- "physchem (lead-like)" --> H(Clustering)
G -- structural alerts --> H

end
end
```
