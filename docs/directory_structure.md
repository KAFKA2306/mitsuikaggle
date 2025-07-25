docs/
├── competition.md
input/
├── target_pairs.csv
├── test.csv
├── train_labels.csv
├── train.csv
├── kaggle_evaluation/
│   ├── __init__.py
│   ├── mitsui_gateway.py
│   ├── mitsui_inference_server.py
│   └── core/
│       ├── __init__.py
│       ├── base_gateway.py
│       ├── kaggle_evaluation.proto
│       ├── relay.py
│       ├── templates.py
│       └── generated/
│           ├── __init__.py
│           ├── kaggle_evaluation_pb2_grpc.py
│           └── kaggle_evaluation_pb2.py
└── lagged_test_labels/
    ├── test_labels_lag_1.csv
    ├── test_labels_lag_2.csv
    ├── test_labels_lag_3.csv
    └── test_labels_lag_4.csv
src/
├── read_project_files.py
└── eda/
    └── eda.py