def extract_silence_level(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        last_line = lines[-1]
        silence_level = last_line.split(":")[-1].strip()
        numeric_value = silence_level.replace("dB", "").strip()
        return numeric_value

# file_path = '/home/kcdh_admin/anindya/Indic-ASR-Transcript-Generator/utils/time_o.txt'
# silence_level = extract_silence_level(file_path)
# print(f"Average silence level: {silence_level}")
