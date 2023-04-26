# Installing

It is strongly advised to use virtual enviroment(s), and install it into one:

```bash
pip install -r requirements.txt
```

# Runnning

Just run the notebooks. Note: the training notebook will run very long as it trains the full network for 50 times.

To launch Jupyter:
```bash
jupyter notebook 
```

# Data preprocessing
Data preprocessing is not the best practice, it is StandardScaled inside the modell execution, with independent scalers for the Validation and Training data. Suprisingli it works like that, but in theory the data should be scaled with the same scaler in the preprocessing stage.
Also the data is casted to `np.float16` to save memory.

# Tensorboard
Tensorboard is set up, you can launch it with the following command, and then open the [http://localhost:8888/](localhost).
```bash
tensorboard --logdir=logs/ --host localhost --port 8888
```

# Results

![Nice graph](/media/epoch_accuracy.png)
