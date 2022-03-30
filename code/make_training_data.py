from os import listdir
import numpy as np
import pickle
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import math
from tqdm.contrib import tzip

class Training_Set:

    def __init__(self, signal_dir, background_dir, split, outdir):
        self.outdir = outdir
        self.signal_dir  = signal_dir
        self.background_dir  = background_dir
        self.files_signal = listdir(signal_dir)
        self.files_background = listdir(background_dir)
        if len(self.files_signal) > len(self.files_background):
            self.total_events = len(self.files_background)
        else:
            self.total_events = len(self.files_signal)
        

        self.test_events = math.floor(float(split) * (self.total_events))

        self.training_set_signal = self.files_signal[self.test_events:self.total_events]
        self.training_set_background = self.files_background[self.test_events:self.total_events]

        self.train_signal_events = np.zeros((len(self.training_set_signal),20,12))
        self.train_background_events = np.zeros((len(self.training_set_signal),20,12))
        self.train_signal_label = np.ones((len(self.training_set_signal),1))
        self.train_background_label =  np.zeros((len(self.training_set_signal),1))
        self.training_data = {}


    def make_training_data(self):
        bin_x = 20
        bin_y = 12

        print("making dataset...")

        for i, sig_event, bkg_event in tzip(range(len(self.training_set_signal)),self.training_set_signal, self.training_set_background):
            signal_hist = np.load(f"{self.signal_dir}/{sig_event}")
            bkg_hist = np.load(f"{self.background_dir}/{bkg_event}")

            self.train_signal_events[i] = Training_Set.reshape_hist(bkg_hist, bin_x, bin_y)
            self.train_background_events[i] = Training_Set.reshape_hist(signal_hist, bin_x, bin_y)
            
    @staticmethod
    def reshape_hist(hist, bin_x, bin_y):
        hist = hist.reshape(bin_x, hist.shape[0]//bin_x, bin_y, hist.shape[1]//bin_y).sum(axis=1).sum(axis=2)
        return hist

    def gather_all(self):
        events = np.concatenate((self.train_signal_events, self.train_background_events))
        labels = np.concatenate((self.train_signal_label, self.train_background_label))
        self.training_data = {"events": events, "labels": labels}
       
    def scale(self):
        self.training_events = self.training_data["events"]
        scale_factor = 1./np.percentile(self.training_events[self.training_events != 0].flatten(), 99.99)
        self.training_data["events"] = self.training_data["events"] * scale_factor
    
    def save(self):
        with open(f"{self.outdir}/training_data.pickle", 'wb') as handle:
           pickle.dump(self.training_data , handle, protocol=3)
        print(f"saved:{self.outdir}")

def main(signal_dir, bkg_dir, split, outdir):
    training_set = Training_Set(signal_dir, bkg_dir, split, outdir)
    training_set.make_training_data()
    training_set.gather_all()
    training_set.scale()
    training_set.save()

        
if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-ds","--signal_dir" , help="Directory containing signal proc histograms from prep_datset.py")
    parser.add_argument("-db","--bkg_dir" , help="Directory containing background proc histograms from prep_datset.py")
    parser.add_argument("-s","--split" ,   help="what fraction of data to use for testing, e.g. 0.25")
    parser.add_argument("-o","--outdir" , help="output directory")
    args = parser.parse_args()
    main(args.signal_dir, args.bkg_dir,args.split, args.outdir)