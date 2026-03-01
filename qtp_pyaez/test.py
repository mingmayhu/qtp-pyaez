import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
import os

WORK_DIR = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
YEARS    = list(range(1979, 2019))

os.chdir(WORK_DIR)

def plot_all_years(filename, title_prefix, output_file, vmin=0, vmax=366):
    n_cols = 8
    n_rows = -(-len(YEARS) // n_cols)  # ceiling division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3.5))
    axes = axes.flatten()

    for i, year in enumerate(YEARS):
        path = f'./data_output/module1/{year}/{filename}'
        arr  = gdal.Open(path).ReadAsArray().astype(float)
        arr[arr < 0] = np.nan  # mask nodata

        im = axes[i].imshow(arr, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[i].set_title(str(year), fontsize=9)
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], shrink=0.8)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    fig.suptitle(title_prefix, fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', dpi=150)
    plt.close()
    print(f'Saved {output_file}')

plot_all_years('LGP New.tif',      'LGP 1979–2018',          './data_output/lgp_all_years.png')
plot_all_years('LGP_Equv New.tif', 'LGP Equivalent 1979–2018', './data_output/lgp_equiv_all_years.png')