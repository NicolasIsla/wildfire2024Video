import argparse
import torch
import pytorch_lightning as pl
from utils import FireDataModule, FireClassifier
from pytorch_lightning.loggers import WandbLogger

# Función para cargar el mejor modelo y correr las métricas de test
def test_model(args):
    # Initialize the DataModule
    data_module = FireDataModule(data_dir=args.data_dir, batch_size=args.batch_size)

    # Cargar el modelo guardado desde el checkpoint
    model = FireClassifier.load_from_checkpoint(args.checkpoint)

    # Inicializa el Trainer de PyTorch Lightning para evaluar el modelo en el conjunto de test
    trainer = pl.Trainer()

    # Ejecuta las métricas de test
    results = trainer.test(model, dataloaders=data_module.test_dataloader())
    
    print(f"Resultados de Test: {results}")

# Función principal con argparse
def main():
    parser = argparse.ArgumentParser(description="Evaluación del clasificador de incendios")

    parser.add_argument('--data_dir', type=str, required=True, help='Directorio de los datos de test')
    parser.add_argument('--checkpoint', type=str, required=True, help='Ruta del checkpoint del mejor modelo guardado')
    parser.add_argument('--batch_size', type=int, default=4, help='Tamaño del lote')

    args = parser.parse_args()
    test_model(args)

if __name__ == '__main__':
    main()
