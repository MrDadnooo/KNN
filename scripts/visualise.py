from io import BytesIO

from matplotlib import pyplot as plt
import matplotlib.patches as patches
from download import dataManager
import numpy as np
import annotation
from parse_xml import Page
from PIL import Image

def plot_text_regions(xml_page: Page, documents: dict[str, annotation.Document], image_uuid: str) -> None:
    fig, ax = plt.subplots()

    ocr_col = np.random.rand(3, )
    bl_col = np.random.rand(3, )
    line_col = np.random.rand(3, )
    for text_region in xml_page:
        polygon = patches.Polygon(text_region.coords, closed=True, fill=False, edgecolor=ocr_col, linewidth=1)
        ax.add_patch(polygon)
        for text_line in text_region:
            polygon = patches.Polygon(text_line.baseline, closed=False, fill=False, edgecolor=bl_col, linewidth=1)
            ax.add_patch(polygon)
            polygon = patches.Polygon(text_line.coords, closed=True, fill=False, edgecolor=line_col, linewidth=1)
            ax.add_patch(polygon)

    doc = documents[image_uuid]
    for im, texts in doc.annotations.items():
        col = np.random.rand(3, )

        if im.ocr_ref:
            image_crop = dataManager.get_image_crops(image_uuid, im.ocr_ref)
            x, y, w, h = (im.ocr_ref.x, im.ocr_ref.y, im.ocr_ref.w, im.ocr_ref.h)
            ax.imshow(Image.open(BytesIO(image_crop.read())), extent=[x, x + w, y, y + h])
        polygon = patches.Polygon(im.coords, closed=True, fill=False, edgecolor=col, linewidth=1)
        ax.add_patch(polygon)
        for text_ann in texts:
            polygon = patches.Polygon(text_ann.coords, closed=True, fill=False, edgecolor=col, linewidth=1)
            ax.add_patch(polygon)

    plt.gca().set_aspect('equal', adjustable='box')
    ax.set_xlim(0, xml_page.width)
    ax.set_ylim(0, xml_page.height)
    plt.axis('off')
    plt.show()
