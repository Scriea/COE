# separate_lines.py

def separate_lines(main_txt_path, odd_txt_path, even_txt_path):
    with open(main_txt_path, 'r') as main_file:
        lines = main_file.readlines()
    
    odd_lines = [line for i, line in enumerate(lines) if i % 2 != 0]
    even_lines = [line for i, line in enumerate(lines) if i % 2 == 0]

    with open(odd_txt_path, 'w') as odd_file:
        odd_file.writelines(odd_lines)
    
    with open(even_txt_path, 'w') as even_file:
        even_file.writelines(even_lines)

# Example usage
main_txt_path = r'D:\Business\transcript_whisper_largev3\telugu_2.txt'
odd_txt_path = r'Telugu_2_Prompt_hyp.txt'
even_txt_path = r'Telugu_2_Test_hyp.txt'

separate_lines(main_txt_path, odd_txt_path, even_txt_path)
