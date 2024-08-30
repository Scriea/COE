import torch
from pyannote.audio import Pipeline


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token="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")
pipeline.to(torch.device(device))

def diarize_audio(filename, num_speakers=None):
    if num_speakers:
        diarization = pipeline(filename, num_speakers=num_speakers)
    else:
        diarization = pipeline(filename)

    actual_result = {}
    for segment in diarization.itersegments():
        speaker_id = diarization.get_labels(segment).pop()
        start, end = segment
        actual_result[(start, end)] = speaker_id

    # print("actual_result: ", actual_result)
    speaker_list = list(actual_result.values())
    updated_speaker_list = swap_elements_by_first_occurrence(speaker_list)
    for idx, key in enumerate(actual_result.keys()):
        actual_result[key] = updated_speaker_list[idx]

    return actual_result

def retrieve_chunks(timestamps, start_time, end_time):
    chunks = []
    for (start, end), speaker in timestamps.items():
        if start_time >= start and end_time <= end:
            chunks.append((round(start_time, 3), round(end_time, 3), speaker))
        elif start_time <= start and end_time >= end:
            chunks.append((round(start, 3), round(end, 3), speaker))
        elif start_time <= start < end_time or start_time < end <= end_time:
            chunks.append((round(max(start_time, start), 3), round(min(end_time, end), 3), speaker))
    return chunks

def swap_all(lst, old_value, new_value):
    swapped_idx = []
    for i, value in enumerate(lst):
        if value == old_value:
            lst[i] = new_value
            swapped_idx.append(i)
        if value == new_value and i not in swapped_idx:
            lst[i] = old_value
            swapped_idx.append(i)
    return lst

def swap_elements_by_first_occurrence(input_list):
    first_occurrence = {}
    my_list_swapped = None
    for i, speaker in enumerate(input_list):
        if speaker not in first_occurrence:
            first_occurrence[speaker] = i

    sorted_keys = sorted(first_occurrence.keys(), key=lambda x: int(x.split('_')[-1]))

    new_index = {sorted_keys[index]: value for index, (key, value) in enumerate(first_occurrence.items())}

    swap_history = {}
    for k, v in new_index.items():
        old_value = input_list[v]
        new_value = k
        if swap_history.get(new_value, None) == old_value:
            continue
        my_list_swapped = swap_all(input_list, old_value, new_value)
        swap_history[old_value] = new_value

    return my_list_swapped