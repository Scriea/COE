import jiwer

def calculate_wer_cer(groundtruth_path, hyp_path):
    # Read the ground truth text
    with open(groundtruth_path, 'r', encoding='utf-8') as file:
        groundtruth_text = file.read().strip()
    
    # Read the hypothesis text
    with open(hyp_path, 'r', encoding='utf-8') as file:
        hyp_text = file.read().strip()
    
    # Calculate WER
    wer = jiwer.wer(groundtruth_text, hyp_text)
    
    # Calculate CER
    cer = jiwer.cer(groundtruth_text, hyp_text)
    
    return wer, cer

# Paths to your ground truth and hypothesis text files
groundtruth_path = '/home/kcdh_admin/anindya/Indic-ASR-Transcript-Generator/utils/samples/groundtruth_transcript.txt' 
hyp_path = '/home/kcdh_admin/anindya/Indic-ASR-Transcript-Generator/hyp_indic_wav2vec2_diarize_20.txt'

wer, cer = calculate_wer_cer(groundtruth_path, hyp_path)
print(f"Word Error Rate (WER): {wer}")
print(f"Character Error Rate (CER): {cer}")
