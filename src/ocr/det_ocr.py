import os
from PIL import Image
import pandas as pd
import numpy as np
import pytesseract
import matplotlib.pyplot as plt
import argparse
from td import TableDetector
from tsr import get_cells_from_rows_cols, get_rows_cols_from_tatr
from utils import *

# Set the TESSDATA_PREFIX environment variable
# os.environ['TESSDATA_PREFIX'] = '/raid/ganesh/vishak/miniconda3/envs/coe/share'


parser = argparse.ArgumentParser(description='Table Detection')
parser.add_argument('-p', '--pdf', type=str, help='Path to the image', required=True)

args = parser.parse_args()

def save_cells(cropped_img, cells, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for row_idx, row in cells.items():
        for col_idx, cell in enumerate(row):
            x1, y1, x2, y2 = cell
            cell_img = cropped_img[y1:y2, x1:x2]
            output_path = os.path.join(output_dir, f'cell_{row_idx}_{col_idx + 1}.png')
            plt.imsave(output_path, cell_img)
            # print(f'Saved {output_path}')


def ocr_cells(cropped_img, cells):
    ocr_data = []
    for row_idx, row in cells.items():
        for col_idx, cell in enumerate(row):
            x1, y1, x2, y2 = cell
            cell_img = cropped_img[y1:y2, x1:x2]
            # Convert cell image to PIL Image for OCR
            cell_pil_img = Image.fromarray(cell_img)
            ocr_result = pytesseract.image_to_string(cell_pil_img, config='--psm 6')
            print(ocr_result)
            ocr_data.append((row_idx, col_idx, ocr_result.strip()))
    return ocr_data




if __name__=="__main__":
    
    table_det = TableDetector()
    images = pdf_to_images(args.pdf)
    image = np.array(images[0])
    # plt.imsave("image.jpg", image)
    dets = table_det.predict(image=image)
    all_ocr_data = []
    for det in dets:
        x1, y1, x2, y2 = map(int, det)  # Convert coordinates to integers
        cropped_img = image[y1:y2, x1:x2]  # Crop the image using the bounding box
        plt.imsave("cropped_img.jpg", cropped_img)
        img_file = "cropped_img.jpg"
        rows, cols = get_rows_cols_from_tatr(img_file)
        print(len(rows))
        print(len(cols))

        rows, cols = order_rows_cols(rows, cols)

        ## Visualize Rows and Columns
        row_image = draw_bboxes(img_file, rows, color = (255, 66, 55), thickness = 2)
        cols_image = draw_bboxes(img_file, cols, color= (22, 44, 255), thickness = 2)
        cv2.imwrite('rows.jpg', row_image)
        cv2.imwrite('cols.jpg', cols_image)


        ## Extracting Cells

        cells = get_cells_from_rows_cols(rows, cols)

        ## Visualize Extracted Cells
        all_cells = []
        for kr in cells.keys():
            all_cells += cells[kr]
        cell_image = draw_bboxes(img_file, all_cells, color = (23, 255, 45), thickness = 1)
        cv2.imwrite('cell.jpg', cell_image)
        # print("Cells: ", cells)
        save_cells(cropped_img, cells, "./cells")

        ## OCR on Cells
        ocr_data = ocr_cells(cropped_img, cells)
        all_ocr_data.extend(ocr_data)

    # Convert OCR data to a pandas DataFrame
    df = pd.DataFrame(all_ocr_data, columns=['Row', 'Column', 'Text'])
    df = df.pivot(index='Row', columns='Column', values='Text')
    print(df)
    # Save DataFrame to a CSV file
    df.to_csv('./extracted_table.csv', index=True)