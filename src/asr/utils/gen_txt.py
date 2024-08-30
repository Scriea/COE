import os
import xml.etree.ElementTree as ET

# Directories for input files
xml_dir = '/raid/ganesh/pdadiga/anindya/Indic-whisper-wav2vec2-Transcription/transcripts/wav2vec2/kathbath_hard_whisper_large/hindi'
audio_dir = '/raid/ganesh/pdadiga/suryansh/Dataset/kathbath/kathbath_hard/kathbath_noisy/hindi/test_known_augmented_audio'

# Directory for output
log_file = '/raid/ganesh/pdadiga/anindya/Indic-whisper-wav2vec2-Transcription/wav2vec2/kathbath_audio/whisper_hard_large.txt'
os.makedirs(os.path.dirname(log_file), exist_ok=True)

# Open log file for writing
with open(log_file, 'w', encoding='utf-8') as log:
    # Process each XML file in the directory
    for xml_file in os.listdir(xml_dir):
        if xml_file.endswith('.xml'):
            print(f'Processing {xml_file}')
            xml_path = os.path.join(xml_dir, xml_file)
            audio_file = os.path.splitext(xml_file)[0][:-7] + '.wav'
            audio_path = os.path.join(audio_dir, audio_file)

            if not os.path.exists(audio_path):
                print(f"Audio file {audio_file} not found for {xml_file}. Skipping.")
                continue

            # Load XML and parse
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Initialize variables for this XML file
            transcript_lines = []

            # Process XML lines
            for line in root.findall('line'):
                words = line.findall('word')
                if words:
                    # Combine words for transcript
                    transcript = ' '.join(word.text if word.text is not None else '' for word in words)
                    transcript_lines.append(transcript)
            
            # Join all transcripts with spaces and write to the log file
            transcript_text = ' '.join(transcript_lines)
            log.write(f"{audio_file}\t{transcript_text}\n")

            print(f"Processed {xml_file}")

print("Processing complete. Results saved in:", log_file)
