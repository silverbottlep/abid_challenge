import os
import json
import numpy as np
import matplotlib.pyplot as plt

metadata_file = "metadata.json"
instance_file = "instances.json"

# getting total quantity across whole datasets
def get_total_quantity(metadata):
    total_quantity = 0
    N = len(metadata)
    for meta in metadata:
        if meta:
            total_quantity += meta['EXPECTED_QUANTITY']
    return total_quantity

def get_max_quantity(metadata):
    max_quantity = 0
    for i in range(0,len(metadata)):
        if metadata[i]:
            quantity = metadata[i]["EXPECTED_QUANTITY"]
            if quantity > max_quantity:
                max_quantity = quantity
    return max_quantity

def show_hist_instance_repeat(instances):
    count = 0
    repeats= np.zeros(len(instances), dtype=int)
    for k in instances:
        v = instances[k]
        repeats[count] = v['repeat']
        count += 1

    n_bins = 51
    repeat_hist, bin_edges = np.histogram(repeats,bins=np.arange(n_bins)+1)
    np.savetxt("repeat_hist.csv",repeat_hist, fmt='%d')
    plt.plot(np.arange(n_bins-1)+1,repeat_hist)
    plt.xlabel('The number of repeatetion')
    plt.ylabel('The number of instances')
    plt.title('Histogram of repeatetion of instances across all bin images')
    plt.show()


def show_hist_quantity(metadata):
    max_quantity = get_max_quantity(metadata)
    quantity_hist = np.zeros(max_quantity+1, dtype=int)
    for i in range(0,len(metadata)):
        if metadata[i]:
            quantity = metadata[i]["EXPECTED_QUANTITY"]
            quantity_hist[quantity] = quantity_hist[quantity] + 1
    np.savetxt("quantity_hist.csv",quantity_hist, fmt='%d')
    plt.plot(quantity_hist)
    plt.xlabel('Quantity in a bin image')
    plt.ylabel('The number of bin images')
    plt.title('Histogram of quantity distribution across all bin images')
    plt.show()

if __name__ == '__main__':
    print("loading metadata...")
    with open(metadata_file) as json_file:
        metadata = json.load(json_file)
    print("loading instance list...")
    with open(instance_file) as json_file:
        instances = json.load(json_file)
    print("loading metadata and instances finished!")

    total_quantity = get_total_quantity(metadata) 
    print("Statistics!")
    print("Total Images: " + str(len(metadata)))
    print("Average expected quantity in a bin: " + str(total_quantity) + "/" 
          + str(len(metadata)) + ": " + str(total_quantity/float(len(metadata))))
    print("The number of instances: " + str(len(instances)))
    
    print("Showing histogram of repeatition")
    show_hist_instance_repeat(instances)
    print("Showing histogram of quantity distribution")
    show_hist_quantity(metadata)
