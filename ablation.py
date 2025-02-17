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
    from pytorch_forecasting.data import GroupNormalizer
    from lightning.pytorch.loggers import TensorBoardLogger
    import lightning.pytorch as pl
    from torch.utils.data import DataLoader


    torch.set_float32_matmul_precision('medium')
    # Load the dataset
    df = pd.read_csv('no_norm_df_trim.csv')

    # Ensure 'time_idx' is sequential and starts from zero for each 'full_sequence_id'
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

    batch_size = 32
    num_workers = 2

    # Create the training dataloader
    train_dataloader = training_set.to_dataloader(train=True, batch_size=batch_size, num_workers=num_workers, drop_last=True)

    # Create the validation dataloader
    val_dataloader = validation_set.to_dataloader(train=False, batch_size=batch_size, num_workers=num_workers, drop_last=True)

    # Create the test dataloader
    test_dataloader = test_set.to_dataloader(train=False, batch_size=batch_size, num_workers=num_workers, drop_last=True)

    total_batches = len(train_dataloader)
    print("total batches: ",total_batches)

    # Configure callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='./ckp',
        filename='tft-test_2-{epoch:02d}-{train_loss:.2f}',
        save_top_k=3,
        mode='min',
        save_last=True,
        every_n_train_steps=5000  # Save a checkpoint every 5,000 training steps
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=8,
        verbose=False,
        mode="min"
    )
    lr_logger = LearningRateMonitor()  # Log the learning rate
    logger = TensorBoardLogger("lightning_logs")  # Log to TensorBoard

    # Configure the trainer
    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="gpu",
        limit_train_batches = 0.4,
        devices=1,  # Set this to the number of GPUs you have
        enable_model_summary=True,
        precision = 32,
        gradient_clip_val=0.6590466012307762,
        callbacks=[lr_logger, early_stop_callback, checkpoint_callback],
        logger=logger,
        log_every_n_steps=100,  # Log metrics every 50 steps
        accumulate_grad_batches=4,
    )

    # Define the Temporal Fusion Transformer model
    tft = TemporalFusionTransformer.from_dataset(
        training_set,
        learning_rate=0.0005,
        hidden_size=32,
        attention_head_size=4,
        dropout=0.3,
        hidden_continuous_size=16,
        loss=QuantileLoss(),
        log_interval=100,                                                                                                        
        reduce_on_plateau_patience=2
    )

    # Train the model
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    # (Optional) Save the model
    trainer.save_checkpoint("tft_group_trim.ckpt")
    # gradient_clip_val': 0.6590466012307762, 'hidden_size': 14,
    #  'dropout': 0.13611191712966667, 'hidden_continuous_size': 14, 'attention_head_size': 2, 'learning_rate': 0.0011509511911692882}