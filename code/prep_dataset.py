import uproot 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import namedtuple
import os
from tqdm import tqdm
import dask as d 
from dask.diagnostics import ProgressBar

def make_plot(histogram, outdir):
    fig = plt.figure(figsize=(12, 10))
    sns.set_palette("pastel")
    sns.set_context("paper", font_scale=0.75) 
    sns.heatmap(histogram.data, xticklabels=histogram.xlabel, yticklabels=histogram.ylabel,cmap="RdBu_r", center=0)
    fig.patch.set_facecolor('white')
    plt.title("Single event distibution")
    fig.show()
    fig.savefig(f"{outdir}/event_histo.png")
    print(f"Saved to : {outdir}")

@d.delayed
def make_dataset(file, directory):
     TH2D = uproot.open(f"{directory}/{file}")['caloGrid']
     TH2D.numpy()
     return np.array(TH2D.values)

def main(directory, outdir, plot, make_data, topology):
    file_list = os.listdir(directory)
    
    TH2D = uproot.open(f"{directory}/{file_list[0]}")['caloGrid']
    TH2D.numpy()
    OneEvent = namedtuple("OneEvent", "data bins xlabel ylabel")
    OneEvent.data   = np.array(TH2D.values)
    OneEvent.bins   = np.array(TH2D.edges, dtype=object)
    OneEvent.ylabel = np.array(TH2D.edges[:][0])
    OneEvent.xlabel = np.array(TH2D.edges[:][1])
    
    if plot:
       make_plot(OneEvent, outdir)
    else:
        pass

    if make_data:
        event_list = []
        print("Buidling DAG...")
        for file in tqdm(file_list):
            event_list.append(make_dataset(file, directory))
        print("Computing...")
        with ProgressBar():
             _ = d.compute(*event_list)
        events = np.asarray(_)
        with open(f"{outdir}/{topology}.npy", 'wb') as f:
            np.save(f, events)
        print(f"Saved as {topology}.npy")
    else:
        pass
   
if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d","--dir" , help="Directory containing histograms")
    parser.add_argument("-o","--outdir" , help="Path to output directory for storage")
    parser.add_argument("-t","--topology" , help="name of process")
    parser.add_argument("-p", "--plot", action="store_true", help="Plot an event?")
    parser.add_argument("-md","--make_data", action="store_true", help="make dataset?")
    args = parser.parse_args()
    main(args.dir, args.outdir, args.plot, args.make_data, args.topology)
