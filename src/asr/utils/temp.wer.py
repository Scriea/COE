import re
import subprocess
from typing import List, Tuple

def parse_file(file_path: str) -> dict:
    parsed_data = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():
                parts = line.split('\t')
                if len(parts) == 2:
                    audio_file_name = re.sub(r'\.[^.]+$', '', parts[0])  # Remove file extension
                    sentence = parts[1].strip()
                    parsed_data[audio_file_name] = sentence
    return parsed_data

def add_spaces(text: str) -> str:
    return ' '.join(text)

def parse_wer_output(output: str) -> float:
    match = re.search(r'WER:\s+([0-9.]+)%', output)
    return float(match.group(1)) / 100 if match else 1.0

def calculate_wer_and_cer_kaldi(hypothesis: dict, ground_truth: dict) -> Tuple[List[Tuple[str, float, float]], float, float]:
    wer_results = []
    cer_results = []
    total_wer = 0.0
    total_cer = 0.0
    count = 0

    for audio_file_name, hyp_sentence in hypothesis.items():
        if audio_file_name in ground_truth:
            gt_sentence = ground_truth[audio_file_name]
            
            with open("hyp.txt", "w") as hyp_file, open("ref.txt", "w") as ref_file:
                hyp_file.write(f'{hyp_sentence}\n')
                ref_file.write(f'{gt_sentence}\n')

            wer_command = ["wer","ref.txt", "hyp.txt"]
            wer_output = subprocess.run(wer_command, capture_output=True, text=True)
            print("WER Command Stdout:", wer_output.stdout)
            print("WER Command Stderr:", wer_output.stderr)  # Print errors
            wer_score = parse_wer_output(wer_output.stdout)
            
            hyp_sentence_with_spaces = add_spaces(hyp_sentence)
            gt_sentence_with_spaces = add_spaces(gt_sentence)
            with open("hyp_cer.txt", "w") as hyp_cer_file, open("ref_cer.txt", "w") as ref_cer_file:
                hyp_cer_file.write(f'{hyp_sentence_with_spaces}\n')
                ref_cer_file.write(f'{gt_sentence_with_spaces}\n')
            cer_command = ["wer", "ref_cer.txt", "hyp_cer.txt"]
            cer_output = subprocess.run(cer_command, capture_output=True, text=True)
            print("CER Command Stdout:", cer_output.stdout)
            print("CER Command Stderr:", cer_output.stderr)  # Print errors
            cer_score = parse_wer_output(cer_output.stdout)

            wer_results.append((audio_file_name, wer_score))
            cer_results.append((audio_file_name, cer_score))
            total_wer += wer_score
            total_cer += cer_score
            count += 1
    
    average_wer = total_wer / count if count > 0 else 0.0
    average_cer = total_cer / count if count > 0 else 0.0
    return wer_results, cer_results, average_wer, average_cer

def main():
    file1_path = '/raid/ganesh/pdadiga/anindya/Indic-whisper-wav2vec2-Transcription/wav2vec2/kathbath_audio/audio.txt'
    file2_path = '/raid/ganesh/pdadiga/suryansh/Dataset/kathbath/kathbath/hindi/test/transcription.txt'
    
    hypothesis_data = parse_file(file1_path)
    ground_truth_data = parse_file(file2_path)
    
    wer_results, cer_results, average_wer, average_cer = calculate_wer_and_cer_kaldi(hypothesis_data, ground_truth_data)
    
    print("WER Results:")
    for audio_file_name, wer_score in wer_results:
        print(f'Audio File: {audio_file_name}, WER: {wer_score:.4f}')
    
    print("\nCER Results:")
    for audio_file_name, cer_score in cer_results:
        print(f'Audio File: {audio_file_name}, CER: {cer_score:.4f}')
    
    print(f'\nAverage WER: {average_wer:.4f}')
    print(f'Average CER: {average_cer:.4f}')

if __name__ == '__main__':
    main()
