import cv2
def draw_bboxes(img_file, bboxes, color = (255, 0, 255), thickness= 2):
    image = cv2.imread(img_file)
    for b in bboxes:
        start_point = (int(b[0]), int(b[1]))
        end_point = (int(b[2]), int(b[3]))
        image = cv2.rectangle(image, start_point, end_point, color, thickness)
    return image