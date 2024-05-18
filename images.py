from PIL import Image
from pillow_heif import register_heif_opener
import os

filepath = "validation/Unknown"

register_heif_opener()
heic_files = [photo for photo in os.listdir(filepath) if '.HEIC' in photo]

for photo in heic_files:
  temp_img = Image.open(os.path.join(filepath, photo))
  png_photo = photo.replace('.HEIC','.png')
  temp_img.save(os.path.join(filepath, png_photo))

  os.remove(os.path.join(filepath, photo))