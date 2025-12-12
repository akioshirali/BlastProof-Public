# BlastProof-Public-

minesweeper-Clean/
│
├── env/
│   ├── minesweeper.py
│   ├── board_generation.py
│   ├── clue.py
│   ├── threebv.py
│   └── sampling.py
│
├── graphs/
│   ├── conventional.py
│   ├── heterograph.py
│   ├── hypergraph.py
│   ├── cached_conventional.py
│   ├── cached_hetero.py
│   ├── cached_hypergraph.py
│   └── utils.py
│
├── models/
│   ├── conventional_gnn.py
│   ├── hetero_gnn.py
│   ├── hypergraph_gnn.py
│   └── policy_network.py
│
├── ppo/
│   ├── buffer.py
│   ├── gae.py
│   ├── ppo_v1.py
│   ├── ppo_v2.py
│   └── utils.py
│
├── curriculum/
│   ├── size_density.py
│   ├── hardness.py
│   ├── fast_curriculum.py
│   ├── stupid_hard.py
│   └── board_sampling.py
│
├── eval/
│   ├── evaluation.py
│   ├── render.py
│   ├── plots.py
│   └── metrics.py
│
├── scripts/
│   ├── train_v1.py
│   ├── train_v2.py
│   ├── evaluate.py
│   └── finetune.py
│
├── tests/
│   ├── test_env.py
│   ├── test_graphs.py
│   ├── test_models.py
│   └── test_ppo.py
│
├── README.md
├── requirements.txt
└── setup.py
