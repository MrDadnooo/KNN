from io import BytesIO

from matplotlib import pyplot as plt
import matplotlib.patches as patches
from download import dataManager
import dataset as ds
import numpy as np
from PIL import Image


def plot_text_regions(data_point: ds.DataPoint) -> None:
    fig, ax = plt.subplots()

    ocr_col = np.random.rand(3, )
    bl_col = np.random.rand(3, )
    line_col = np.random.rand(3, )
    for text_region in data_point.page:
        polygon = patches.Polygon(text_region.coords, closed=True, fill=False, edgecolor=ocr_col, linewidth=1)
        ax.add_patch(polygon)
        for text_line in text_region:
            polygon = patches.Polygon(text_line.baseline, closed=False, fill=False, edgecolor=bl_col, linewidth=1)
            ax.add_patch(polygon)
            polygon = patches.Polygon(text_line.coords, closed=True, fill=False, edgecolor=line_col, linewidth=1)
            ax.add_patch(polygon)

    for im in data_point.img_annotations:
        col = np.random.rand(3, )

        if im.ocr_ref:
            image_crop = dataManager.get_image_crops(data_point.page.uuid, im.ocr_ref)
            x, y, w, h = (im.ocr_ref.x, im.ocr_ref.y, im.ocr_ref.w, im.ocr_ref.h)
            ax.imshow(Image.open(BytesIO(image_crop.read())), extent=[x, x + w, y, y + h])
        polygon = patches.Polygon(im.coords, closed=True, fill=False, edgecolor=col, linewidth=1)
        ax.add_patch(polygon)
        for text_ann in im.texts:
            polygon = patches.Polygon(text_ann.coords, closed=True, fill=False, edgecolor=col, linewidth=1)
            ax.add_patch(polygon)

    plt.gca().set_aspect('equal', adjustable='box')
    ax.set_xlim(0, data_point.page.width)
    ax.set_ylim(0, data_point.page.height)
    plt.axis('off')
    plt.show()
