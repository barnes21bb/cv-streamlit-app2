import cv2
import xml.etree.ElementTree as ET


def draw_annotations(image, annotations):
    """Draw bounding boxes on an image"""
    img_with_boxes = image.copy()

    colors = {
        'good-cup': (0, 255, 0),
        'bad-cup': (255, 0, 0),
        'no-cup': (255, 255, 0)
    }

    for ann in annotations:
        x1, y1, x2, y2 = ann['bbox']
        class_name = ann['class']
        color = colors.get(class_name, (0, 0, 255))

        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)

        label = f"{class_name}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale,
                                                       thickness)
        cv2.rectangle(img_with_boxes,
                      (x1, y1 - text_height - 4),
                      (x1 + text_width + 4, y1),
                      color, -1)
        cv2.putText(img_with_boxes, label, (x1 + 2, y1 - 2), font,
                    font_scale, (255, 255, 255), thickness)

    return img_with_boxes


def generate_pascal_voc_xml(annotations_dict, video_name, video_shape):
    """Return PASCAL VOC XML strings per frame."""
    output = {}
    h, w, c = video_shape

    for frame_num, frame_annotations in annotations_dict.items():
        if not frame_annotations:
            continue

        annotation = ET.Element("annotation")

        folder = ET.SubElement(annotation, "folder")
        folder.text = "frames"

        filename = ET.SubElement(annotation, "filename")
        filename.text = f"{video_name}_frame_{frame_num}.jpg"

        source = ET.SubElement(annotation, "source")
        database = ET.SubElement(source, "database")
        database.text = "Custom Video Annotation"

        size = ET.SubElement(annotation, "size")
        width = ET.SubElement(size, "width")
        width.text = str(w)
        height = ET.SubElement(size, "height")
        height.text = str(h)
        depth = ET.SubElement(size, "depth")
        depth.text = str(c)

        segmented = ET.SubElement(annotation, "segmented")
        segmented.text = "0"

        for ann in frame_annotations:
            obj = ET.SubElement(annotation, "object")

            name = ET.SubElement(obj, "name")
            name.text = ann['class']

            pose = ET.SubElement(obj, "pose")
            pose.text = "Unspecified"

            truncated = ET.SubElement(obj, "truncated")
            truncated.text = "0"

            difficult = ET.SubElement(obj, "difficult")
            difficult.text = "0"

            bndbox = ET.SubElement(obj, "bndbox")
            xmin = ET.SubElement(bndbox, "xmin")
            xmin.text = str(ann['bbox'][0])
            ymin = ET.SubElement(bndbox, "ymin")
            ymin.text = str(ann['bbox'][1])
            xmax = ET.SubElement(bndbox, "xmax")
            xmax.text = str(ann['bbox'][2])
            ymax = ET.SubElement(bndbox, "ymax")
            ymax.text = str(ann['bbox'][3])

        from xml.dom import minidom
        rough_string = ET.tostring(annotation, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="  ")

        lines = pretty_xml.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        pretty_xml = '\n'.join(non_empty_lines)

        output[frame_num] = pretty_xml

    return output
