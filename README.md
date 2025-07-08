# MoE_LHC
This project investiagets the use of mixture of experts to perform anomaly detection on high dimensional data at the LHC.

Each expert is a MLP module with access to a subset of the input features (the subsets are motivated by domain knowledge), reducing the challenges of course of dimensionality.

A trainable gate module combines the experts outputs in the final output.

We studied two configurations:
- **Global gate**: the gate is composed of `n_experts` trainable parameters. They apply to all data.
- **Local gate**: the is a MLP module with `n_experts` neurons in the output layer. It takes as input all the features and softmax activates and can weigth experts differently according to the input event.

This repository cointains code for both configurations.
To run a trainig:
- set up the problem in the `run_toys_[local/gloabl].py`
- run
  ```
  python run_toys_[local/gloabl].py -p toy_[local/global]_kfold_sigmoid.py -t [NR. TOYS] -l [LOCAL JOB] -s [SEED]
  ```

NOTE: some parameters of the training task (for instance, the MLPs architectures) are set in the `toy_[local/global]_kfold_sigmoid.py` file and not configurable in the `run_toys_[local/gloabl].py` file. If it's more convenient, you can modify the code to make this more flexible.
