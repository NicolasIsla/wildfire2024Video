import gdown
import zipfile
import os

def download_and_extract(url, destination):
    # Extraer el ID del archivo desde el enlace de Google Drive
    file_id = url.split("/d/")[1].split("/")[0]
    download_url = f"https://drive.google.com/uc?id={file_id}"
    
    # Descargar el archivo
    gdown.download(download_url, destination, quiet=False)

    # Extraer el archivo ZIP
    with zipfile.ZipFile(destination, 'r') as zip_ref:
        zip_ref.extractall(os.path.dirname(destination))
    os.remove(destination)  # Eliminar el archivo ZIP después de extraer

if __name__ == "__main__":
    # Ruta donde se guardarán los archivos descargados
    path = "/data/nisla"
    if not os.path.exists(path):
        os.makedirs(path)

    # Nombre del archivo que será descargado
    destination = os.path.join(path, 'fire_dataset.zip')

    # URL del archivo de Google Drive
    url = "https://drive.google.com/file/d/1siZ7m9QYzD2NUrAf3PhKzlSZqWpcQ_5D/view?usp=drive_link"

    # Descargar y extraer
    download_and_extract(url, destination)

    print("Download and extraction completed.")
