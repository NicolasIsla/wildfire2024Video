import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from utils import FireDataModule, FireClassifier
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# Entrenar el modelo
def train_model(args):
    # Initialize the DataModule
    data_module = FireDataModule(data_dir=args.data_dir, batch_size=args.batch_size)

    # Initialize the model
    model = FireClassifier(learning_rate=args.learning_rate, lstm_layers=args.lstm_layers)

    # Define the checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val_f1", 
        mode="max", 
        save_top_k=4, 
        filename='{epoch:02d}-{val_f1:.2f}',  # Custom filename with epoch and val_acc
    )

    # Initialize the WandbLogger
    wandb_logger = WandbLogger(project='lstm-efficentnet', log_model=True)

    # Initialize the Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback],
        logger=wandb_logger
    )

    # Train the model
    trainer.fit(model, data_module)

    # # Once training is complete, run testing with the best checkpoint
    # best_model_path = checkpoint_callback.best_model_path
    # print(f"Best model saved at: {best_model_path}")

    # # Load the best checkpoint and test the model
    # trainer.test(ckpt_path=best_model_path, datamodule=data_module)

# Función principal con argparse
def main():
    parser = argparse.ArgumentParser(description="Entrenamiento de clasificador de incendios")

    parser.add_argument('--data_dir', type=str, required=True, help='Directorio de los datos de entrenamiento')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Tasa de aprendizaje')
    parser.add_argument('--epochs', type=int, default=50, help='Número de épocas de entrenamiento')
    parser.add_argument('--batch_size', type=int, default=32, help='Tamaño del lote')
    parser.add_argument('--lstm_layers', type=int, default=2, help='Número de capas LSTM')

    args = parser.parse_args()
    train_model(args)

if __name__ == '__main__':
    main()