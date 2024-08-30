import os
from jiwer import wer, cer

def get_ground_truth_filename(hyp_filename):
    return hyp_filename.replace('_hyp', '')

def calculate_error_rates(hyp_folder, groundtruth_folder):
    for hyp_filename in os.listdir(hyp_folder):
        if hyp_filename.endswith('.txt'):
            hyp_path = os.path.join(hyp_folder, hyp_filename)
            groundtruth_filename = get_ground_truth_filename(hyp_filename)
            groundtruth_path = os.path.join(groundtruth_folder, groundtruth_filename)
            
            if os.path.exists(groundtruth_path):
                with open(hyp_path, 'r') as hyp_file, open(groundtruth_path, 'r') as gt_file:
                    hyp_text = hyp_file.read().strip()
                    groundtruth_text = gt_file.read().strip()
                    
                    wer_score = wer(groundtruth_text, hyp_text)
                    cer_score = cer(groundtruth_text, hyp_text)
                    
                    print(f"For file {hyp_filename}:")
                    print(f"  Word Error Rate (WER): {wer_score}")
                    print(f"  Character Error Rate (CER): {cer_score}")
                    print()

# Example usage
hyp_folder = r'/raid/ganesh/pdadiga/anindya/Indic-whisper-wav2vec2-Transcription/utils/transcript_whisper_medium/hyp'
groundtruth_folder = r'/raid/ganesh/pdadiga/anindya/Indic-whisper-wav2vec2-Transcription/utils/transcript_whisper_medium/groundtruth'

calculate_error_rates(hyp_folder, groundtruth_folder)
