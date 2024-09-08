import torch
import numpy as np
from PIL import Image
import random
import glob
import os
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, Precision, Recall, F1
import torchvision.models as models
import pytorch_lightning as pl

class FireSeriesDataset(Dataset):
    def __init__(self, root_dir, img_size=112, frames=32, position=16, transform=None):
        self.img_size = img_size
        self.frames = frames
        self.position = position
        self.transform = transform

        # Buscar todos los directorios de videos
        self.videos = sorted(glob.glob(f"{root_dir}/*"))

        # Almacenar los frames seleccionados para cada video
        self.selected_frames = []

        for video_dir in self.videos:
            # Obtener todos los frames y etiquetas en el directorio del video
            frames = sorted(glob.glob(f"{video_dir}/*.jpg"))  # o .png según corresponda
            labels = sorted(glob.glob(f"{video_dir}/*.txt"))

            # Verificar si todos los frames tienen o no tienen anotaciones
            all_frames_have_annotations = all(os.path.getsize(label) > 0 for label in labels)
            no_frames_have_annotations = all(os.path.getsize(label) == 0 for label in labels)

            if no_frames_have_annotations:
                # Asegurarse de que haya suficientes frames
                if len(frames) >= self.frames:
                    start_idx = random.randint(0, len(frames) - self.frames)
                    selected_frames = frames[start_idx:start_idx + self.frames]
                else:
                    selected_frames = frames  # Tomamos todos los disponibles
                selected_labels = [0] * len(selected_frames)  # Etiqueta 0 para todos
            elif all_frames_have_annotations:
                # Asegurarse de que haya suficientes frames
                if len(frames) >= self.frames:
                    start_idx = random.randint(0, len(frames) - self.frames)
                    selected_frames = frames[start_idx:start_idx + self.frames]
                else:
                    selected_frames = frames  # Tomamos todos los disponibles
                selected_labels = [1] * len(selected_frames)  # Etiqueta 1 para todos
            else:
                # Encontrar la posición del primer archivo de etiqueta no vacío
                first_non_empty_index = None
                for i, label in enumerate(labels):
                    if os.path.getsize(label) > 0:  # Si el archivo tiene contenido
                        first_non_empty_index = i
                        break

                if first_non_empty_index is not None:
                    start_idx = max(0, first_non_empty_index - self.position)
                    end_idx = min(len(frames), start_idx + self.frames)

                    selected_frames = frames[start_idx:end_idx]
                    selected_labels = labels[start_idx:end_idx]

                    # Verificar si se seleccionó menos de `self.frames`
                    if len(selected_frames) < self.frames:
                        remaining_frames = frames[end_idx:end_idx + (self.frames - len(selected_frames))]
                        selected_frames.extend(remaining_frames)
                        selected_labels.extend([0] * len(remaining_frames))  # Etiqueta 0 para los frames restantes
                else:
                    selected_frames = []
                    selected_labels = []

            # Rellenar si es necesario con los frames siguientes, sin rellenar con `None`
            while len(selected_frames) < self.frames:
                selected_frames.append(frames[-1])  # Repetimos el último frame disponible si es necesario
                selected_labels.append(0)  # Etiqueta 0

            self.selected_frames.append((selected_frames, selected_labels))

    def __len__(self):
        return len(self.selected_frames)

    def generate_random_bounding_box(self, w, h):
        """Genera una bounding box aleatoria dentro de los límites de la imagen"""
        xc = random.uniform(0.5, 0.9)  # Coordenada X central aleatoria entre 0.25 y 0.75
        yc = random.uniform(0.5, 0.9)  # Coordenada Y central aleatoria entre 0.25 y 0.75
        wb = random.uniform(0.05, 0.15)  # Ancho de la caja aleatorio entre 10% y 50% del ancho de la imagen
        hb = random.uniform(0.05, 0.15)  # Altura de la caja aleatoria entre 10% y 50% del alto de la imagen
        return [xc, yc, wb, hb]

    def __getitem__(self, idx):
        img_paths, label_paths = self.selected_frames[idx]

        labels = []
        bounding_boxes = []

        # Revisar si alguno de los frames tiene anotación
        no_annotations = True  # Asumimos que no hay anotaciones al principio
        for label_file in label_paths:
            if label_file is not None and os.path.getsize(label_file) > 0:
                no_annotations = False
                break
        
        if no_annotations:
            # En lugar de intentar abrir una imagen, creamos una imagen vacía
            w, h = self.img_size, self.img_size
            random_bbox = self.generate_random_bounding_box(w, h)
            bounding_boxes = [random_bbox for _ in range(self.frames)]  # Usar la misma bounding box aleatoria para todos los frames
            labels = [0] * self.frames  # Etiquetas 0 para todos los frames
        else:
            # Si hay anotaciones, procesar los frames normalmente
            for label_file in label_paths:
                if label_file is None or os.path.getsize(label_file) == 0:
                    labels.append(0)  # Etiqueta vacía
                    bounding_boxes.append([0, 0, 0, 0])  # Bounding box vacía
                else:
                    with open(label_file, "r") as f:
                        lines = f.readlines()
                    if len(lines) > 0:
                        box = np.array(lines[0].split(" ")[1:5]).astype("float")
                        labels.append(1)  # Hay contenido
                        bounding_boxes.append(box)
                    else:
                        labels.append(0)  # Etiqueta vacía
                        bounding_boxes.append([0, 0, 0, 0])  # No hay bounding box

        # Recortar las imágenes según la bounding box más grande o aleatoria
        images = []
        for file in img_paths:
            if file is None:
                # Crear una imagen en blanco si no hay suficientes frames
                blank_image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
                img = Image.fromarray(blank_image)
            else:
                img = Image.open(file)
            
            images.append(img)

        # Asegurar que las imágenes tienen la misma dimensión
        if len(images) == 0:
            return None, None

        w, h = images[0].size

        bounding_boxes = np.array(bounding_boxes)
        xc = np.median(bounding_boxes[:, 0])
        yc = np.median(bounding_boxes[:, 1])
        wb = np.max(bounding_boxes[:, 2])
        hb = np.max(bounding_boxes[:, 3])

        # Determinar el tamaño del recorte
        crop_size = max(wb * w, hb * h)
        if crop_size < self.img_size:
            crop_size = self.img_size

        x0 = int(xc * w - crop_size / 2)
        y0 = int(yc * h - crop_size / 2)
        x1 = int(xc * w + crop_size / 2)
        y1 = int(yc * h + crop_size / 2)

        # Recortar y redimensionar las imágenes
        img_list = []
        for im in images:
            cropped_image = im.crop((x0, y0, x1, y1))
            cropped_image = cropped_image.resize((self.img_size, self.img_size))
            img_list.append(cropped_image)

        # Aplicar transformaciones
        if self.transform:
            tensor_list = [self.transform(img) for img in img_list]
        else:
            tensor_list = [torch.tensor(np.array(img)) for img in img_list]

        # Concatenar las imágenes
        tensor_list = torch.stack(tensor_list)

        return tensor_list, torch.tensor(labels)
    
import os
import random
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl

class FireDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=16, img_size=112, num_workers=8):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_workers = num_workers

        # Definir transformaciones para train y val
        self.train_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=(-10, 10)),
            transforms.ColorJitter(brightness=(0.9, 1.1), contrast=(0.9, 1.1), saturation=(0.9, 1.1), hue=(-0.1, 0.1)),
            transforms.RandomResizedCrop(self.img_size, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def setup(self, stage=None):
        self.train_dataset = FireSeriesDataset(
            os.path.join(self.data_dir, "train"), self.img_size, transform=self.train_transform
        )
        self.val_dataset = FireSeriesDataset(
            os.path.join(self.data_dir, "val"), self.img_size, transform=self.val_transform
        )
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    

class FireClassifier(pl.LightningModule):
    def __init__(self, learning_rate=1e-4):
        super(FireClassifier, self).__init__()
        self.save_hyperparameters()

        # Usamos una ResNet como extractor de características.
        # Pretrained sobre ImageNet, usualmente se carga con 3 canales.
        resnet = models.resnet50(pretrained=True)
        # Removemos la capa final para usarla como extractor de características.
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])

        # LSTM que procesará las características extraídas.
        # Número de características de la salida del último bloque conv de ResNet.
        num_features = resnet.fc.in_features
        self.lstm = nn.LSTM(input_size=32768, hidden_size=256, batch_first=True, num_layers=3)

        # Capa de clasificación.
        self.classifier = nn.Linear(256, 1)  # Salida binaria

        # Dropout para regularización
        self.dropout = nn.Dropout(0.2)

        # Métricas
        self.train_accuracy = Accuracy(task="binary")
        self.val_accuracy = Accuracy(task="binary")
        self.train_precision = Precision(task="binary")
        self.val_precision = Precision(task="binary")
        self.train_recall = Recall(task="binary")
        self.val_recall = Recall(task="binary")
        # f1
        self.train_f1 = F1(task="binary")
        self.val_f1 = F1(task="binary")

    def forward(self, x):
        # x shape: [batch_size, seq_length, channels, height, width]
        # Procesa cada imagen de la secuencia a través del extractor de características.
        batch_size, seq_length, C, H, W = x.size()
        x = x.view(batch_size * seq_length, C, H, W)
        x = self.feature_extractor(x)

        # Reformatear salida para la LSTM
        x = x.view(batch_size, seq_length, -1)

        # Pasar las características por la LSTM
        x, _ = self.lstm(x)

        # Aplicamos la capa de clasificación a cada frame de la secuencia
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch

        # y tiene la forma [batch_size, seq_length], por lo que necesitamos ajustarlo
        y_hat = self(x).squeeze()

        # Aplicar la pérdida sobre todos los frames
        loss = F.binary_cross_entropy_with_logits(y_hat, y.float())
        acc = self.train_accuracy(torch.sigmoid(y_hat), y.int())
        precision = self.train_precision(torch.sigmoid(y_hat), y.int())
        recall = self.train_recall(torch.sigmoid(y_hat), y.int())
        # f1
        f1 = self.train_f1(torch.sigmoid(y_hat), y.int())
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        self.log("train_precision", precision)
        self.log("train_recall", recall)
        self.log("train_f1", f1)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()

        # Aplicar la pérdida sobre todos los frames
        loss = F.binary_cross_entropy_with_logits(y_hat, y.float())
        acc = self.val_accuracy(torch.sigmoid(y_hat), y.int())
        precision = self.val_precision(torch.sigmoid(y_hat), y.int())
        recall = self.val_recall(torch.sigmoid(y_hat), y.int())
        # f1
        f1 = self.val_f1(torch.sigmoid(y_hat), y.int())

        self.log("val_loss", loss)
        self.log("val_acc", acc)
        self.log("val_precision", precision)
        self.log("val_recall", recall)
        self.log("val_f1", f1)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams['learning_rate'], weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }
