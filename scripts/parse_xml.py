from typing import IO, Iterator

import paramiko
import numpy as np
import xml.etree.ElementTree as ET
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib
import annotation

from scripts import annotation


class TextLine:
    def __init__(self, index, custom_heights, coords, baseline, conf, text):
        self.index = index
        self.custom_heights = tuple(float(num) for num in custom_heights.split(":")[-1][1:-1].split(","))
        self.coords = parse_points(coords)
        self.baseline = parse_points(baseline)
        self.conf = conf
        self.text = text


class TextRegion:
    def __init__(self, coords, text_lines):
        self.coords = parse_points(coords)
        self.text_lines = text_lines

    def __iter__(self) -> Iterator[TextLine]:
        return iter(self.text_lines)


class Page:
    def __init__(self, uuid, width, height, text_regions):
        self.uuid = uuid
        self.width = int(width)
        self.height = int(height)
        self.text_regions = text_regions

    def __iter__(self) -> Iterator[TextRegion]:
        return iter(self.text_regions)


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
        print(im.coords)
        polygon = patches.Polygon(im.coords, closed=True, fill=False, edgecolor=col, linewidth=1)
        ax.add_patch(polygon)
        for text_ann in texts:
            polygon = patches.Polygon(text_ann.coords, closed=True, fill=False, edgecolor=col, linewidth=1)
            ax.add_patch(polygon)


    plt.gca().set_aspect('equal', adjustable='box')
    ax.set_xlim(0, xml_page.width)
    ax.set_ylim(0, xml_page.height)
    plt.show()


def parse_points(input_str: str) -> np.array:
    """
    Convert coordination like string to a 2D numpy array of x,y coordinates
    :param input_str: string coordinates
    :return:
    """
    return np.array([list(map(int, point.split(','))) for point in input_str.split()])


def parse_xml_document(xml_file: IO[bytes]) -> Page:
    def tag(el): return el.tag.split('}', 1)[1] if '}' in el.tag else el.tag

    # convert the XML document to a tree-like structure
    xml_root = ET.parse(xml_file)
    # find the <Page/> element (that element contains all data we're interested in)
    page_el = None
    for child in xml_root.getroot():
        if tag(child) == 'Page':
            page_el = child
            break
    # TODO maybe trow a different exception
    if not page_el:
        raise NotImplemented

    width = page_el.get('imageWidth')
    height = page_el.get('imageHeight')
    uuid = page_el.get('imageFilename')

    text_regions: list[TextRegion] = []
    for text_region_el in page_el:
        text_lines = []
        coords = None
        for text_line_el in text_region_el:
            if tag(text_line_el) == 'Coords':
                coords = text_line_el.get('points')
            elif tag(text_line_el) == 'TextLine':
                index = text_line_el.get('index')
                custom_heights = text_line_el.get('custom')
                text_line_coords = None
                baseline = None
                conf = None
                text = None
                for els in text_line_el:
                    if tag(els) == 'Coords':
                        text_line_coords = els.get('points')
                    elif tag(els) == 'Baseline':
                        baseline = els.get('points')
                    elif tag(els) == 'TextEquiv':
                        conf = els.get('conf')
                        for unicode_el in els:
                            text = unicode_el.text
                text_lines.append(TextLine(index, custom_heights, text_line_coords, baseline, conf, text))
        text_regions.append(TextRegion(coords, text_lines))
    # create the top-level element representing whole xml page and return
    return Page(uuid, width, height, text_regions)



