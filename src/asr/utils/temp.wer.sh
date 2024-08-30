#!/bin/bash

# Function to parse the input file and store it in an associative array
parse_file() {
    declare -A parsed_data
    while IFS=$'\t' read -r audio_file sentence; do
        if [[ -n $audio_file && -n $sentence ]]; then
            # Remove file extension
            audio_file_name="${audio_file%.*}"
            parsed_data["$audio_file_name"]="$sentence"
        fi
    done < "$1"
    echo "${parsed_data[@]}"
}

# Function to add spaces between characters in a string
add_spaces() {
    echo "$1" | sed 's/\(.\)/\1 /g'
}

# Function to calculate WER and CER
calculate_wer_and_cer() {
    declare -A hypothesis
    declare -A ground_truth
    hypothesis=($(parse_file "$1"))
    ground_truth=($(parse_file "$2"))

    total_wer=0
    total_cer=0
    count=0

    for audio_file_name in "${!hypothesis[@]}"; do
        if [[ -n "${ground_truth[$audio_file_name]}" ]]; then
            hyp_sentence="${hypothesis[$audio_file_name]}"
            gt_sentence="${ground_truth[$audio_file_name]}"

            # Write the sentences to temp files for WER calculation
            echo "$gt_sentence" > /tmp/gt.txt
            echo "$hyp_sentence" > /tmp/hyp.txt

            # Calculate WER
            wer_score=$(wer /tmp/gt.txt /tmp/hyp.txt | grep -oP '\d+\.\d+')

            # Calculate CER by adding spaces
            hyp_sentence_with_spaces=$(add_spaces "$hyp_sentence")
            gt_sentence_with_spaces=$(add_spaces "$gt_sentence")

            echo "$gt_sentence_with_spaces" > /tmp/gt.txt
            echo "$hyp_sentence_with_spaces" > /tmp/hyp.txt

            cer_score=$(wer /tmp/gt.txt /tmp/hyp.txt | grep -oP '\d+\.\d+')

            # Store results
            echo "Audio File: $audio_file_name, WER: $wer_score, CER: $cer_score"
            total_wer=$(echo "$total_wer + $wer_score" | bc)
            total_cer=$(echo "$total_cer + $cer_score" | bc)
            count=$((count + 1))
        fi
    done

    if [[ $count -gt 0 ]]; then
        average_wer=$(echo "$total_wer / $count" | bc -l)
        average_cer=$(echo "$total_cer / $count" | bc -l)
    else
        average_wer=0
        average_cer=0
    fi

    echo -e "\nAverage WER: $average_wer"
    echo "Average CER: $average_cer"
}

# Main function
main() {
    # Paths to the hypothesis and ground truth files
    file1_path='/raid/ganesh/pdadiga/anindya/Indic-whisper-wav2vec2-Transcription/wav2vec2/kathbath_audio/audio.txt'
    file2_path='/raid/ganesh/pdadiga/suryansh/Dataset/kathbath/kathbath/hindi/test/transcription.txt'

    # Calculate WER and CER
    calculate_wer_and_cer "$file1_path" "$file2_path"
}

# Run the main function
main
