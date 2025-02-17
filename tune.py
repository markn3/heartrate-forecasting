def lr_tune():
    # find optimal learning rate
    from lightning.pytorch.tuner import Tuner

    res = Tuner(trainer).lr_find(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        max_lr=10.0,
        min_lr=1e-6,
    )

    print(f"suggested learning rate: {res.suggestion()}")
    fig = res.plot(show=True, suggest=True)
    fig.show()

def hyperparam_tune(train_dataloader, val_dataloader):
    import pickle

    from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
    # https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.models.temporal_fusion_transformer.tuning.optimize_hyperparameters.html#pytorch_forecasting.models.temporal_fusion_transformer.tuning.optimize_hyperparameters
    # create study
    study = optimize_hyperparameters(
        train_dataloader,
        val_dataloader,
        model_path="optuna_test",
        n_trials=50,
        max_epochs=25,
        gradient_clip_val_range=(0.01, 1.0),
        hidden_size_range=(8, 64),
        hidden_continuous_size_range=(8, 64),
        attention_head_size_range=(1, 4),
        learning_rate_range=(0.001, 0.01),
        dropout_range=(0.1, 0.3),
        trainer_kwargs=dict(limit_train_batches=20),
        reduce_on_plateau_patience=4,
        use_learning_rate_finder=False,  # use Optuna to find ideal learning rate or use in-built learning rate finder
        verbose=2
    )

    # save study results - also we can resume tuning at a later point in time
    with open("test_study.pkl", "wb") as fout:
        pickle.dump(study, fout)

    # show best hyperparameters
    print(study.best_trial.params)
    # {'gradient_clip_val': 0.09051868072403757, 'hidden_size': 23, 'dropout': 0.20556764609524497, 'hidden_continuous_size': 9, 'attention_head_size': 2, 'learning_rate': 0.002281159739080524}

 

if __name__ == "__main__":

    import pandas as pd
    import numpy as np
    import torch
    import gc

    from pytorch_forecasting.models import TemporalFusionTransformer
    from pytorch_lightning import trainer
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.metrics import MAE, QuantileLoss
    from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import TensorBoardLogger
    import lightning.pytorch as pl
    from torch.utils.data import DataLoader
    from pytorch_forecasting.data import GroupNormalizer



    torch.set_float32_matmul_precision('medium')
    # Load the dataset
    df = pd.read_csv('no_norm_df_trim.csv')

    # Ensure 'time_idx' is sequential and starts from zero for each 'full_sequence_id'
    # Assuming 'time' is your timestamp column; adjust if necessary
    df.sort_values(['full_sequence_id', 'charttime'], inplace=True)
    df['time_idx'] = df.groupby('full_sequence_id').cumcount()

    print(df)

    # Convert boolean columns to strings
    bool_columns = [
        "ele", "dir_em", "amb_obs",
        # Add other boolean disease category columns here
        "blood", "circulatory", "congenital", "digestive", "endocrine",
        "genitourinary", "infectious", "injury", "mental", "neoplasms", "perinatal",
        "respiratory", "is_male"
    ]
    df[bool_columns] = df[bool_columns].astype(str)

    for col in bool_columns:
        df[col] = df[col].astype('category')

    # Split data by 'full_sequence_id' to prevent data leakage
    unique_ids = df['full_sequence_id'].unique()
    np.random.shuffle(unique_ids)

    train_size = int(len(unique_ids) * 0.7)
    val_size = int(len(unique_ids) * 0.15)

    train_ids = unique_ids[:train_size]
    val_ids = unique_ids[train_size:train_size + val_size]
    test_ids = unique_ids[train_size + val_size:]

    train_data = df[df['full_sequence_id'].isin(train_ids)]
    val_data = df[df['full_sequence_id'].isin(val_ids)]
    test_data = df[df['full_sequence_id'].isin(test_ids)]

    # After data processing
    del df
    gc.collect()

    # Define maximum encoder and prediction lengths
    max_encoder_length = 18
    max_prediction_length = 6

    training_set = TimeSeriesDataSet(
        train_data,
        time_idx="time_idx",
        target="future_heart_rate",
        group_ids=["full_sequence_id"],
        max_encoder_length=18,
        max_prediction_length=6,
        static_categoricals=[
            "ele",
            "dir_em",
            "amb_obs",
            "is_male",
            # Add the disease categories (booleans)
            "blood", "circulatory", "congenital", "digestive", "endocrine",
            "genitourinary", "infectious", "injury", "mental", "neoplasms", "perinatal",
            "respiratory",
        ],
        static_reals=["los", "anchor_age"],  # Include static numerical features
        time_varying_known_reals=["time_idx"],  # Features known at all timestamps
        time_varying_unknown_reals=[
            "heart_rate", "systolic",
            "respiratory_rate", "oxygen_saturation", "temperature_f"
        ],
        target_normalizer=GroupNormalizer(
        groups=["full_sequence_id"],  # Normalize target by patient group
        method="standard",  # Standard scaling
        transformation="softplus"  # Ensure positive outputs
        ),
        scalers={
        "time_varying_unknown_reals": GroupNormalizer(groups=["full_sequence_id"]),
        "static_reals": GroupNormalizer(groups=["full_sequence_id"]),
        },
        allow_missing_timesteps=True
    )

    validation_set = TimeSeriesDataSet(
        val_data,
        time_idx="time_idx",
        target="future_heart_rate",
        group_ids=["full_sequence_id"],
        max_encoder_length=18,
        max_prediction_length=6,
        static_categoricals=[
            "ele",
            "dir_em",
            "amb_obs",
            "is_male",
            # Add the disease categories (booleans)
            "blood", "circulatory", "congenital", "digestive", "endocrine",
            "genitourinary", "infectious", "injury", "mental", "neoplasms", "perinatal",
            "respiratory",
        ],
        static_reals=["los", "anchor_age"],  # Include static numerical features
        time_varying_known_reals=["time_idx"],  # Features known at all timestamps
        time_varying_unknown_reals=[
            "heart_rate", "systolic",
            "respiratory_rate", "oxygen_saturation", "temperature_f"
        ],
        target_normalizer=GroupNormalizer(
        groups=["full_sequence_id"],  # Normalize target by patient group
        method="standard",  # Standard scaling
        transformation="softplus"  # Ensure positive outputs
        ),
        scalers={
        "time_varying_unknown_reals": GroupNormalizer(groups=["full_sequence_id"]),
        "static_reals": GroupNormalizer(groups=["full_sequence_id"]),
        },
        allow_missing_timesteps=True
    )

    test_set = TimeSeriesDataSet(
        test_data,
        time_idx="time_idx",
        target="future_heart_rate",
        group_ids=["full_sequence_id"],
        max_encoder_length=18,
        max_prediction_length=6,
        static_categoricals=[
            "ele",
            "dir_em",
            "amb_obs",
            "is_male",
            # Add the disease categories (booleans)
            "blood", "circulatory", "congenital", "digestive", "endocrine",
            "genitourinary", "infectious", "injury", "mental", "neoplasms", "perinatal",
            "respiratory",
        ],
        static_reals=["los", "anchor_age"],  # Include static numerical features
        time_varying_known_reals=["time_idx"],  # Features known at all timestamps
        time_varying_unknown_reals=[
            "heart_rate", "systolic",
            "respiratory_rate", "oxygen_saturation", "temperature_f"
        ],
        target_normalizer=GroupNormalizer(
        groups=["full_sequence_id"],  # Normalize target by patient group
        method="standard",  # Standard scaling
        transformation="softplus"  # Ensure positive outputs
        ),
        scalers={
        "time_varying_unknown_reals": GroupNormalizer(groups=["full_sequence_id"]),
        "static_reals": GroupNormalizer(groups=["full_sequence_id"]),
        },
        allow_missing_timesteps=True
    )
    batch_size = 16
    num_workers = 2

    # Create the training dataloader
    train_dataloader = training_set.to_dataloader(train=True, batch_size=batch_size, num_workers=num_workers)

    # Create the validation dataloader
    val_dataloader = validation_set.to_dataloader(train=False, batch_size=batch_size, num_workers=num_workers)

    # Create the test dataloader
    test_dataloader = test_set.to_dataloader(train=False, batch_size=batch_size, num_workers=num_workers)

    total_batches = len(train_dataloader)
    print("total batches: ",total_batches)

    # Configure callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='./ckp',
        filename='tft-checkpoint-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
        save_last=True,
        every_n_train_steps=1000  # Save a checkpoint every 5,000 training steps
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=10,
        verbose=False,
        mode="min"
    )
    lr_logger = LearningRateMonitor()  # Log the learning rate
    logger = TensorBoardLogger("lightning_logs")  # Log to TensorBoard

    # Define the loss function
    quantile_loss = QuantileLoss(quantiles=[0.1, 0.5, 0.9])


    # Configure the trainer
    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="gpu",
        limit_train_batches=0.5,
        devices=1,  # Set this to the number of GPUs you have
        enable_model_summary=True,
        precision = 32,
        gradient_clip_val=0.1,
        callbacks=[lr_logger, early_stop_callback, checkpoint_callback],
        logger=logger,
        log_every_n_steps=10,  # Log metrics every 50 steps
    )

    # Define the Temporal Fusion Transformer model
    tft = TemporalFusionTransformer.from_dataset(
        training_set,
        learning_rate=0.01,
        hidden_size=16,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=8,
        loss=quantile_loss,
        log_interval=100,
        reduce_on_plateau_patience=4
    )

    hyperparam_tune(train_dataloader, val_dataloader)