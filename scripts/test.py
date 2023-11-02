from pathlib import Path
from PIL import Image

imgs_kstr = [Image.open(path) for path in sorted(Path("../img_kstr").glob("*.jpg"))]
print(imgs_kstr)