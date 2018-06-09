import Augmentor
import os

path_to_data = "/Users/Khmer/Developer/HWR2018/CharacterRecognition/monkbrill_171005_jpg/"
path_to_output = "/Users/Khmer/Developer/HWR2018/CharacterRecognition/monkbrill_171005_jpg_data_augmented/"
try:
    os.stat(path_to_output)
except:
    os.mkdir(path_to_output)

print(os.listdir(path_to_data))
# Create a pipeline
p = Augmentor.Pipeline(path_to_data,path_to_output)
p.crop_random(probability=0.5,percentage_area=0.9)
p.gaussian_distortion(probability=0.9,grid_width=6,grid_height=6,magnitude=4,corner='bell',method='in')

num_of_samples = int(1e4)

p.sample(num_of_samples)
