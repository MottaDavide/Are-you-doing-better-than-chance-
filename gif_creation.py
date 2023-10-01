import imageio.v2 as imageio
from pathlib import Path

# Load images
path = Path("/Users/mottad/Library/CloudStorage/OneDrive-LuxotticaGroupS.p.A/Desktop")



from PIL import Image
frames = [Image.open(image) for image in  [path / 'plot_uniform.png', path /'plot_beta.png', path /'plot_exp.png']]
frame_one = frames[0]
frame_one.save(path / "plot.gif", format="GIF", append_images=frames,
            save_all=True, duration=1000, loop=0)
