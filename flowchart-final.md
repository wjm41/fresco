# Flowchart for FRESCO code - v2

```mermaid
flowchart TD
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
