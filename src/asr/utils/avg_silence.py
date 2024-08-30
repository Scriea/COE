def calculate_average_silence_duration(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()[2:]  # Skip the first two lines

    silences = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 3:
            start_silence = float(parts[0])
            end_silence = float(parts[1])
            duration = end_silence - start_silence
            silences.append(duration)

    if silences:
        average_duration = min(silences)
        return average_duration
    else:
        return 0


file_path = r"mp3splt.log"  
average_silence_duration = calculate_average_silence_duration(file_path)
print(f"Average Silence Duration: {average_silence_duration:.2f} seconds")
