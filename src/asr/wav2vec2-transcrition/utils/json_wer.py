import json
import re
import subprocess

def load_json(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def load_txt(txt_file):
    with open(txt_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        return {line.split('\t')[0]: line.split('\t')[1].strip() for line in lines}

def add_spaces(text: str) -> str:
    """
    Add spaces between characters in the text.
    """
    return ' '.join(text)

def parse_wer_output(output: str) -> float:
    """
    Parse the output of the 'wer' command to extract the WER score.
    """
    match = re.search(r'WER:\s+([0-9.]+)%', output)
    return float(match.group(1)) / 100 if match else 1.0

def compute_wer(ground_truth, hypothesis):
    # Write the ground truth and hypothesis to temporary files
    with open("ref.txt", "w") as ref_file, open("hyp.txt", "w") as hyp_file:
        ref_file.write(f'{ground_truth}\n')
        hyp_file.write(f'{hypothesis}\n')

    # Run the 'wer' command
    wer_command = ["wer", "ref.txt", "hyp.txt"]
    wer_output = subprocess.run(wer_command, capture_output=True, text=True)
    print("WER Command Stdout:", wer_output.stdout)
    print("WER Command Stderr:", wer_output.stderr)  # Print errors
    
    # Parse the WER output
    return parse_wer_output(wer_output.stdout)

def compute_cer(ground_truth, hypothesis):
    """
    Compute CER by adding spaces to the ground truth and hypothesis texts
    and then computing WER on these space-separated texts.
    """
    ground_truth_with_spaces = add_spaces(ground_truth)
    hypothesis_with_spaces = add_spaces(hypothesis)
    return compute_wer(ground_truth_with_spaces, hypothesis_with_spaces)

def main():
    json_file = '/raid/ganesh/pdadiga/suryansh/Dataset/kathbath/kathbath_hard/kathbath_noisy/hindi/test_known_augmented.json'
    txt_file = '/raid/ganesh/pdadiga/anindya/Indic-whisper-wav2vec2-Transcription/wav2vec2/kathbath_audio/wave2vec_kathbath_hard.txt'
    
    # Load data from files
    json_data = load_json(json_file)
    txt_data = load_txt(txt_file)

    # Initialize lists to store WER and CER
    wer_list = []
    cer_list = []

    for entry in json_data:
        audio_file = entry['audio_filepath'].split('/')[-1]
        ground_truth_text = entry['text']
        
        # Find corresponding hypothesis text
        hypothesis_text = txt_data.get(audio_file, None)
        if hypothesis_text:
            # Compute WER
            wer_value = compute_wer(ground_truth_text, hypothesis_text)
            wer_list.append(wer_value)
            
            # Compute CER
            cer_value = compute_cer(ground_truth_text, hypothesis_text)
            cer_list.append(cer_value)
            
            print(f"File: {audio_file}\nGround Truth: {ground_truth_text}\nHypothesis: {hypothesis_text}\nWER: {wer_value:.4f}\nCER: {cer_value:.4f}\n")

    # Compute and print average WER and CER
    if wer_list:
        average_wer = sum(wer_list) / len(wer_list)
        print(f"Average WER: {average_wer:.4f}")
    else:
        print("No matching files found for WER")

    if cer_list:
        average_cer = sum(cer_list) / len(cer_list)
        print(f"Average CER: {average_cer:.4f}")
    else:
        print("No matching files found for CER")

if __name__ == "__main__":
    main()
