# Handwriting recognition

Project for course handwriting recognition

### Running instruction

1. Copy the images to Input folder

2. (Optional) Modify path and suffix in Preprocess/preprocess.py (if the suffix of images is not pgm)

3. Run the command below using Python 3; if you see 'Successfully process image xxx', which means the command is successful

    ```
    python3 Preprocess/preprocess.py
    ```
    
5. Run the command below using Python 3; if you see 'Recognition done for image xxx', which means the recognition is successful
    ```
    python3 CharacterRecognition/run_recognition.py
    ```
    
6. Check the output text files in current folder