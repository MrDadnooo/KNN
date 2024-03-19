from shapely.geometry import Polygon
import parse_xml as OCR
import annotation


class Dataset:
    def __init__(self):
        ...


def test(ocr_data: OCR.Page, ann_el: annotation.Document):
    matched_annotations: dict[annotation.Annotation, tuple[int, OCR.TextLine]] = {}
    for image, text_anns in ann_el.annotations.items():
        for text_ann in text_anns:
            curr_best = (0.0, None)
            for text_region in ocr_data:
                for text_line in text_region:
                    ocr_poly = Polygon(text_line.coords)
                    ann_poly = Polygon(text_ann.coords)
                    intersect = ocr_poly.intersection(ann_poly)
                    if intersect.area > curr_best[0]:
                        curr_best = (intersect.area, text_line)
            matched_annotations[text_ann] = curr_best
    ...
