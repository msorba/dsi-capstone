import matplotlib.pyplot as plt
import pylab

def plot_and_save(image_arr,og_filename, suffix="wrinkle"):
    plt.figure(num=None, figsize=(10, 10), dpi=80)
    plt.imshow(image_arr, cmap='viridis')

    filename = 'mysite/static/results/%s_%s.png' % (og_filename.split('/')[-1].split('.')[0],suffix)
    pylab.savefig(filename,bbox_inches='tight')