'''
Script to randomly augment input data in histogram frame 
by applying random rotations and flips.
'''
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from numpy import expand_dims
from keras.preprocessing.image import ImageDataGenerator
import random
import math

def clean_data(element):
    element = element.reshape(element.shape[0],element.shape[1],1)
    element=expand_dims(element,0)
    return element
    
def apply_flip(data):
    one_event = clean_data(data)
    flips = [ImageDataGenerator(horizontal_flip=True), ImageDataGenerator(vertical_flip=True)]
    im = random.choice(flips).flow(x=one_event, batch_size=1,shuffle=False)
    histo = im.next()
    histo = histo.reshape(data.shape[0],data.shape[1])
    return histo

def apply_rotation(data):
    one_event = clean_data(data)
    angle = random.randint(0,360)
    datagen = ImageDataGenerator(rotation_range=angle)
    im = datagen.flow(x=one_event, batch_size=1,shuffle=False)
    histo = im.next()
    histo = histo.reshape(data.shape[0],data.shape[1])
    return histo

def main(data, outdir, topology):
    data = np.load(data)
    rotated, flipped = [], []
    for_rotation = np.random.randint(0,math.floor(data.shape[0]/2), size=math.floor(data.shape[0]/2))
    for_flip = np.random.randint((math.floor(data.shape[0])/2), math.floor(data.shape[0]), size=math.floor(data.shape[0]/2))
    for index_rot, index_flip in zip(for_rotation, for_flip):
        rotated.append(apply_rotation(data[:][index_rot]))
        flipped.append(apply_flip(data[:][index_flip]))

    data_out = np.array(rotated + flipped)
    with open(f"{outdir}/{topology}.npy", 'wb') as f:
        np.save(f, data_out)
        print(f"Saved as {topology}_augmented.npy")
    
if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d","--data" , help="Directory containing histograms")
    parser.add_argument("-o","--outdir" , help="Path to output directory for storage")
    parser.add_argument("-t","--topology" , help="name of process")
    args = parser.parse_args()
    main(args.data, args.outdir, args.topology)