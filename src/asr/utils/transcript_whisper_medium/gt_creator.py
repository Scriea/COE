import os
import pandas as pd

def process_csv_folder(csv_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(csv_folder):
        if filename.endswith('.csv'):
            csv_path = os.path.join(csv_folder, filename)
            df = pd.read_csv(csv_path)
            
            if 'file_name' in df.columns and 'ground_truth' in df.columns:
                sorted_df = df.sort_values(by='file_name')
                
                txt_filename = os.path.splitext(filename)[0] + '.txt'
                txt_path = os.path.join(output_folder, txt_filename)
                
                with open(txt_path, 'w') as txt_file:
                    for ground_truth in sorted_df['ground_truth']:
                        txt_file.write(str(ground_truth) + '\n')

# Example usage
csv_folder = r'D:\Business\transcript_whisper_largev3\Wav2vec2 Output'
output_folder = r'groundtruth'

process_csv_folder(csv_folder, output_folder)
