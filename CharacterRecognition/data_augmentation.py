import Augmentor
import os

path_to_data = "/Users/Khmer/Developer/HWR2018/CharacterRecognition/monkbrill_171005_jpg/"
path_to_output = "/Users/Khmer/Developer/HWR2018/CharacterRecognition/monkbrill_171005_jpg_data_augmented_500000_all/"
try:
    os.stat(path_to_output)
except:
    os.mkdir(path_to_output)

print(os.listdir(path_to_data))
# Create a pipeline
p = Augmentor.Pipeline(path_to_data,path_to_output)
p.crop_random(probability=0.3,percentage_area=0.95)
p.gaussian_distortion(probability=0.9,grid_width=8,grid_height=8,magnitude=5,corner='bell',method='in')
p.black_and_white(probability=1.0, threshold=64)
p.random_erasing(probability=0.5,rectangle_area=0.3)

num_of_samples = int(50e4)

p.sample(num_of_samples)
