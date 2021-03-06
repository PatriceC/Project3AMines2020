# Projet 3A Mines : Prédiction de la diversité de populations de bactéries par analyse du graphe d’interaction

Il existe différentes méthodes pour essayer de prédire la diversité des bactéries qui seront présentes après un certain temps : simulation in vitro, in silico… Pour cette dernière, on teste in vitro l’interaction entre deux populations de bactéries et on observe un score d’inhibition de croissance entre une population A et une population B. Ce score est ensuite utilisé dans des
algorithmes simulant le comportement des diverses populations. On peut ainsi obtenir des courbes, montrant la croissance des différentes populations de bactéries en fonction du temps.
La façon dont les différentes bactéries interagissent deux à deux peut être modélisée sous forme d’un graphe pondéré orienté. Les nœuds sont les familles de bactéries et un arc de A vers B de poids P signifie que les bactéries A inhibent les bactéries B avec le score d’inhibition P.
Le but de ce projet est d’utiliser des méthodes de fouille de graphes pour prédire la diversité dans les populations de bactéries. Un premier travail de bibliographie et d’expérimentation a été fait par Tristan Durey.

On va dans un premier temps s'intéresser au [GCN](https://arxiv.org/pdf/1609.02907.pdf "SEMI-SUPERVISED CLASSIFICATION WITH
GRAPH CONVOLUTIONAL NETWORKS") pour prédire des classes de comportements de bactéries