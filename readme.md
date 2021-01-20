# DouDiZhu

Action-Prediction Monte-Carlo Tree Search for DouDiZhu self-play reinforcement learning.

## requirements

```python
Cython==0.29.20 
torch==1.3.0
paramiko==2.7.2
```

## compile

We use Cython to implement the DouDiZhu game rules and the Monte-Carlo Tree Search algorithm, so we should compile Cython code fisrt.

```shell
cd game
python compile.py build_ext --inplace
cd ..
cd node
python compile.py build_ext --inplace
```

## self-play data generation

```shell
python gen.py
```

You can change the hyperparameters in main function of gen.py, we show them below.

```python
config.set_device_ids([0, 1, 2, 3, 4, 5, 6, 7])  # GPU used to generated self-play data.
process_per_device = 3  # number of process per GPU
number = 64  # number of games per process
```

The self-play data that we generated will be placed at path data/gen/.

## train

```shell
python train.py
```

Please put the data to data/train and data/test before we start the training process. The neural network parameters will be saved as .pkl file. The default file name and path is save/model.pkl.

Hyperparameters are mainly in config.py.