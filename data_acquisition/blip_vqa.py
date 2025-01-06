import torch
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
import time, os
import cv2
import numpy as np

def is_fading_to_black(frame, threshold=30):
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate the mean intensity of the frame
    mean_intensity = np.mean(gray_frame)
    
    # Check if the mean intensity is below a certain threshold
    # The threshold should be close to 0 (black)
    return mean_intensity < threshold

processor = BlipProcessor.from_pretrained("ybelkada/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("ybelkada/blip-vqa-base", torch_dtype=torch.float16).to("cuda")

images = [(Image.open(f'../keyframes/' + kf), kf) for kf in sorted(os.listdir('../keyframes')) if kf.endswith('.png')]

start_time = time.time()

keyframes_to_remove = set()
answers = {}

question = "Does the photo show a house interior?"
for idx, (raw_image, name) in enumerate(images):
    inputs = processor(raw_image, question, return_tensors="pt").to("cuda", torch.float16)
    out = model.generate(**inputs)
    answer = processor.decode(out[0], skip_special_tokens=True)
    if answer == "no":
        answers[name] = answer
        continue
    curr_image = cv2.imread(f'../keyframes/' + name)
    if is_fading_to_black(curr_image):
        answers[name] = "fading to black"
        continue
    answers[name] = answer
    

for idx, (keyframe, answer) in enumerate(answers.items()):
    if answer == "fading to black":
        keyframes_to_remove.add(keyframe)
    elif answer == "no":
        #Check if the last keyframe was also a no
        if idx / len(answers) < 0.1 or idx / len(answers) > 0.9:
            keyframes_to_remove.add(keyframe)
            continue
        if idx >= 1 and (images[idx-1][1] in keyframes_to_remove): 
            keyframes_to_remove.add(keyframe)
#Traverse the dictionary reversed
for idx, (keyframe, answer) in enumerate(reversed(answers.items())):
    if answer == "no":
        #Check if the last keyframe was also a no
        if idx / len(answers) < 0.1 or idx / len(answers) > 0.9:
            keyframes_to_remove.add(keyframe)
            continue
        if idx >= 1 and (images[-idx][1] in keyframes_to_remove):
            keyframes_to_remove.add(keyframe)

# Add outliers
for idx, (keyframe, answer) in enumerate(answers.items()):
    if answer == "yes":
        #Check if the last and next keyframe were also a no 
        if idx >= 1 and idx < len(answers) - 1 and (answers[images[idx-1][1]] == "no" and answers[images[idx+1][1]] == "no"):
            keyframes_to_remove.add(keyframe)
        

print(f"Keyframes to remove: {sorted(keyframes_to_remove)}")
print(f"Answers len: {len(answers)}")

for kf in keyframes_to_remove:
    os.remove(f'../keyframes/' + kf)

print(f"Elapsed time: {time.time() - start_time:.2f} seconds")