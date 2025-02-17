if __name__ == "__main__":

    import numpy as np
    import pandas as pd
    import torch
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_lightning import Trainer
    import gc
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import torch
    import gc
    import io
    from pytorch_forecasting.models import TemporalFusionTransformer
    from pytorch_lightning import trainer
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.metrics import MAE, QuantileLoss
    from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import TensorBoardLogger
    import lightning.pytorch as pl
    from torch.utils.data import DataLoader
    # Load your data
    df = pd.read_csv('final_dataframe.csv')

    # Ensure 'time_idx' is sequential and starts from zero for each 'full_sequence_id'
    df.sort_values(['full_sequence_id', 'charttime'], inplace=True)
    df['time_idx'] = df.groupby('full_sequence_id').cumcount()

    # Convert boolean columns to strings and categories
    bool_columns = [
        "urg", "sd_adm", "oms_ad", "ew_em", "obs", "ele",
        "dir_obs", "dir_em", "amb_obs",
        "blood", "circulatory", "congenital", "digestive", "endocrine",
        "genitourinary", "infectious", "injury", "mental", "misc",
        "muscular", "neoplasms", "nervous", "pregnancy", "perinatal",
        "respiratory", "skin", "is_male"
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

    # Clean up
    del df
    gc.collect()

    # Define maximum encoder and prediction lengths
    max_encoder_length = 18
    max_prediction_length = 6
    sequence_length = max_encoder_length + max_prediction_length

    # Define batch size
    batch_size = 16

    # Function to trim datasets
    def trim_dataset(data, group_id_col, sequence_length, batch_size):
        # Create a TimeSeriesDataSet to calculate the number of samples
        temp_dataset = TimeSeriesDataSet(
            data,
            time_idx="time_idx",
            target="future_heart_rate",
            group_ids=[group_id_col],
            max_encoder_length=max_encoder_length,
            max_prediction_length=max_prediction_length,
            static_categoricals=[
                "urg", "sd_adm", "oms_ad", "ew_em", "obs", "ele",
                "dir_obs", "dir_em", "amb_obs", "is_male",
                "blood", "circulatory", "congenital", "digestive", "endocrine",
                "genitourinary", "infectious", "injury", "mental", "misc",
                "muscular", "neoplasms", "nervous", "pregnancy", "perinatal",
                "respiratory", "skin",
            ],
            static_reals=["los", "anchor_age", "max_seq_num"],
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_reals=[
                "heart_rate", "systolic", "diastolic",
                "mean", "respiratory_rate", "oxygen_saturation",
                "pulse_oximetry", "temperature_f"
            ],
            allow_missing_timesteps=True
        )
        
        total_samples = len(temp_dataset)
        print(f"Total number of samples before trimming: {total_samples}")
        
        # Calculate the number of full batches
        num_full_batches = total_samples // batch_size
        required_samples = num_full_batches * batch_size
        samples_to_remove = total_samples - required_samples
        print(f"Samples to remove: {samples_to_remove}")
        
        if samples_to_remove == 0:
            # No trimming needed
            return data
        
        # Get unique group_ids
        group_ids = data[group_id_col].unique()
        
        # Shuffle group_ids to randomize
        np.random.shuffle(group_ids)
        
        # Initialize variables
        samples_count = total_samples
        removed_group_ids = []
        
        # Remove group_ids until the total samples align with batch_size
        for gid in group_ids:
            gid_data = data[data[group_id_col] == gid]
            gid_length = len(gid_data)
            
            # Calculate number of samples (sequences) contributed by this group_id
            sequence_count = max(0, gid_length - max_encoder_length - max_prediction_length + 1)
            
            if sequence_count == 0:
                continue  # Skip if this group_id doesn't contribute any samples
            
            samples_count -= sequence_count
            removed_group_ids.append(gid)
            
            if samples_count % batch_size == 0:
                break
        
        # Remove the selected group_ids from the data
        trimmed_data = data[~data[group_id_col].isin(removed_group_ids)]
        
        # Verify the new total number of samples
        temp_dataset = TimeSeriesDataSet(
            trimmed_data,
            time_idx="time_idx",
            target="future_heart_rate",
            group_ids=[group_id_col],
            max_encoder_length=max_encoder_length,
            max_prediction_length=max_prediction_length,
            static_categoricals=[
                "urg", "sd_adm", "oms_ad", "ew_em", "obs", "ele",
                "dir_obs", "dir_em", "amb_obs", "is_male",
                "blood", "circulatory", "congenital", "digestive", "endocrine",
                "genitourinary", "infectious", "injury", "mental", "misc",
                "muscular", "neoplasms", "nervous", "pregnancy", "perinatal",
                "respiratory", "skin",
            ],
            static_reals=["los", "anchor_age", "max_seq_num"],
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_reals=[
                "heart_rate", "systolic", "diastolic",
                "mean", "respiratory_rate", "oxygen_saturation",
                "pulse_oximetry", "temperature_f"
            ],
            allow_missing_timesteps=True
        )
        
        new_total_samples = len(temp_dataset)
        print(f"Total number of samples after trimming: {new_total_samples}")
        assert new_total_samples % batch_size == 0, "Total samples not divisible by batch_size after trimming"
        
        return trimmed_data

    # Trim training data
    trimmed_train_data = trim_dataset(train_data, 'full_sequence_id', sequence_length, batch_size)

    # Trim validation data
    trimmed_val_data = trim_dataset(val_data, 'full_sequence_id', sequence_length, batch_size)

    # Trim test data
    trimmed_test_data = trim_dataset(test_data, 'full_sequence_id', sequence_length, batch_size)

    # Create TimeSeriesDataSet objects
    training_set = TimeSeriesDataSet(
        trimmed_train_data,
        time_idx="time_idx",
        target="future_heart_rate",
        group_ids=["full_sequence_id"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=[
            "urg", "sd_adm", "oms_ad", "ew_em", "obs", "ele",
            "dir_obs", "dir_em", "amb_obs", "is_male",
            "blood", "circulatory", "congenital", "digestive", "endocrine",
            "genitourinary", "infectious", "injury", "mental", "misc",
            "muscular", "neoplasms", "nervous", "pregnancy", "perinatal",
            "respiratory", "skin",
        ],
        static_reals=["los", "anchor_age", "max_seq_num"],
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=[
            "heart_rate", "systolic", "diastolic",
            "mean", "respiratory_rate", "oxygen_saturation",
            "pulse_oximetry", "temperature_f"
        ],
        allow_missing_timesteps=True
    )

    validation_set = TimeSeriesDataSet(
        trimmed_val_data,
        time_idx="time_idx",
        target="future_heart_rate",
        group_ids=["full_sequence_id"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=[
            "urg", "sd_adm", "oms_ad", "ew_em", "obs", "ele",
            "dir_obs", "dir_em", "amb_obs", "is_male",
            "blood", "circulatory", "congenital", "digestive", "endocrine",
            "genitourinary", "infectious", "injury", "mental", "misc",
            "muscular", "neoplasms", "nervous", "pregnancy", "perinatal",
            "respiratory", "skin",
        ],
        static_reals=["los", "anchor_age", "max_seq_num"],
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=[
            "heart_rate", "systolic", "diastolic",
            "mean", "respiratory_rate", "oxygen_saturation",
            "pulse_oximetry", "temperature_f"
        ],
        allow_missing_timesteps=True
    )

    test_set = TimeSeriesDataSet(
        trimmed_test_data,
        time_idx="time_idx",
        target="future_heart_rate",
        group_ids=["full_sequence_id"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=[
            "urg", "sd_adm", "oms_ad", "ew_em", "obs", "ele",
            "dir_obs", "dir_em", "amb_obs", "is_male",
            "blood", "circulatory", "congenital", "digestive", "endocrine",
            "genitourinary", "infectious", "injury", "mental", "misc",
            "muscular", "neoplasms", "nervous", "pregnancy", "perinatal",
            "respiratory", "skin",
        ],
        static_reals=["los", "anchor_age", "max_seq_num"],
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=[
            "heart_rate", "systolic", "diastolic",
            "mean", "respiratory_rate", "oxygen_saturation",
            "pulse_oximetry", "temperature_f"
        ],
        allow_missing_timesteps=True
    )

    # Create DataLoaders
    num_workers = 2  # Adjust as needed

    train_dataloader = training_set.to_dataloader(
        train=True,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True
    )

    val_dataloader = validation_set.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True
    )

    test_dataloader = test_set.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True
    )


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

    # Configure the trainer
    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="cpu",
        # limit_train_batches = 0.2,
        devices=1,  # Set this to the number of GPUs you have
        enable_model_summary=True,
        precision = 32,
        gradient_clip_val=0.1,
        callbacks=[lr_logger, early_stop_callback, checkpoint_callback],
        logger=logger,
        log_every_n_steps=5,  # Log metrics every 50 steps
    )

    # Define the Temporal Fusion Transformer model
    tft = TemporalFusionTransformer.from_dataset(
        training_set,
        learning_rate=0.001,
        hidden_size=16,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=8,
        loss=QuantileLoss(),
        log_interval=100,
        reduce_on_plateau_patience=4
    )

    print("Loading checkpoint...")
    checkpoint_path = './ckp/tft-checkpoint-epoch=10-val_loss=0.10.ckpt'
    with open(checkpoint_path, 'rb') as f:
        checkpoint_buffer = io.BytesIO(f.read())
    print("Loaded checkpoint")

    # Load the model from the buffer
    print("Loading model")
    model_loaded = TemporalFusionTransformer.load_from_checkpoint(checkpoint_buffer)
    print("Model loaded")

    # Calculate mean absolute error on validation set
    results = model_loaded.predict(
        val_dataloader,
        mode="raw",
        return_x=True,
        return_y=True,  # Ensure targets are included
        trainer_kwargs=dict(accelerator="cpu")
    )

    # Unpack the tuple returned by predict
    predictions, x, y, index, decoder_lengths = results

    # Debugging
    print(f"Type of predictions: {type(predictions)}")
    print(f"Shape of predictions[0]: {predictions[0].shape if isinstance(predictions, list) else 'Not a list'}")
    print(f"Keys in x: {x.keys() if hasattr(x, 'keys') else 'x is not a dict'}")

    # Extract targets
    if 'decoder_target' in x:
        targets = x['decoder_target']
        print(f"Shape of targets: {targets.shape}")
    else:
        raise KeyError("'decoder_target' not found in x")

    # Extract the median predictions
    median_idx = model_loaded.loss.quantiles.index(0.5)
    median_predictions = predictions[0][:, :, median_idx]

    # Compute MAE
    from pytorch_forecasting.metrics import MAE

    mae_metric = MAE()
    mae = mae_metric(median_predictions, targets)
    print(f"Mean Absolute Error: {mae}")

    # Generate raw predictions for plotting
    raw_results = model_loaded.predict(
        val_dataloader,
        mode="raw",  # Get raw predictions including quantiles
        return_x=True,  # Include input data for plotting
        trainer_kwargs=dict(accelerator="cpu")
    )

    # Unpack results
    predictions, x, y, index, decoder_lengths = results

    # Plot the first 10 examples
    for idx in range(10):  # Adjust the range as needed
        fig = model_loaded.plot_prediction(
            x=raw_results.x,                        # Input data
            out=raw_results.output,                 # Full predictions tensor wrapped in a dictionary
            idx=idx,                                # Index of the example to plot
            add_loss_to_title=True                  # Add loss value to the plot title
        )
        plt.show()  # Display the figure

    interpretation = model_loaded.interpret_output(raw_results.output, reduction="sum")
    fig = model_loaded.plot_interpretation(interpretation)
    plt.show()