from typing import IO, Iterator

import numpy as np
import xml.etree.ElementTree as eTree
import translator
from shapely import Polygon


class TextLine:
    def __init__(self, index, custom_heights, coords, baseline, conf, text):
        self.index = index
        self.custom_heights = tuple(float(num) for num in custom_heights.split(":")[-1][1:-1].split(","))
        self.coords = parse_points(coords)
        self.baseline = parse_points(baseline)
        self.conf = conf
        self.text = text
        self.en_text = translator.translate(text)
        self.text_region = None


class TextRegion:
    def __init__(self, coords, text_lines):
        self.coords = parse_points(coords)
        self.text_lines = text_lines
        self.page = None

    def __iter__(self) -> Iterator[TextLine]:
        return iter(self.text_lines)


class Page:
    def __init__(self, uuid: str, width, height, text_regions):
        self.uuid = uuid[5:] if uuid.startswith("uuid:") else uuid
        self.width = int(width)
        self.height = int(height)
        self.text_regions = text_regions

    def __iter__(self) -> Iterator[TextRegion]:
        return iter(self.text_regions)


def parse_points(input_str: str) -> np.array:
    """
    Convert coordination like string to a 2D numpy array of x,y coordinates
    :param input_str: string coordinates
    :return:
    """
    return np.array([list(map(int, point.split(','))) for point in input_str.split()])


def parse_xml_document(xml_file: IO[bytes], ann_rec) -> Page:
    def tag(el): return el.tag.split('}', 1)[1] if '}' in el.tag else el.tag

    # convert the XML document to a tree-like structure
    xml_root = eTree.parse(xml_file)
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
        text_region = TextRegion(coords, text_lines)

        # if the current region overlaps with any image it is ignored
        overlapping: bool = False
        for img_ann in list(ann_rec.annotations.keys()):
            img_poly = Polygon(img_ann.coords)
            reg_poly = Polygon(np.array([list(map(int, point.split(','))) for point in coords.split()]))
            i_sect = reg_poly.intersection(img_poly)
            if i_sect.area >= reg_poly.area - reg_poly.area * 0.05:
                overlapping = True
                break

        if not overlapping:
            text_regions.append(text_region)
            for text_line in text_region:
                text_line.text_region = text_region
    # create the top-level element representing whole xml page and return
    page = Page(uuid, width, height, text_regions)
    for text_region in page:
        text_region.page = page
    return page



