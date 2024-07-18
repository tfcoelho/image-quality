from matplotlib import pyplot as plt
import os
import numpy as np
import math
import SimpleITK as sitk
from typing import Optional
from exclusion_lists import rumc_exclude_subject_ids_hbv_scan_quality

list_scan_ids = rumc_exclude_subject_ids_hbv_scan_quality
sequences_wanted = ['hbv']  # the sequences wanted should have the codes present in this list
pathname = '/Volumes/pelvis/data/prostate-MRI/rumc/images'


def _get_list_middle_slices(path: str, list_scan_ids: list, sequences_wanted: list) -> list:
    """
    Retrieve the middle slices from the specified scans and sequences.

    Args:
        path (str): The root directory containing the patient data.
        list_scan_ids (list): A list of scan IDs to process.
        sequences_wanted (list): A list of sequences to consider.

    Returns:
        list: A list of middle slices from the specified scans and sequences.
    """
    list_middle_slices = []

    for scan_id in list_scan_ids:
        try:
            patient_id, study_id = scan_id.split('_')
            patient_path = os.path.join(path, patient_id)

            if not os.path.exists(patient_path):
                raise FileNotFoundError(f"Patient directory not found: {patient_path}")

            for scan in os.listdir(patient_path):
                if scan.startswith(scan_id):
                    for sequence in sequences_wanted:
                        if sequence in scan:
                            scan_path = os.path.join(patient_path, scan)

                            try:
                                # Read MHA image
                                image = sitk.ReadImage(scan_path)
                                # Get image array
                                image_array = sitk.GetArrayFromImage(image)
                                middle_slice_idx = np.shape(image_array)[0] // 2
                                # Get middle slice and add to list
                                list_middle_slices.append(image_array[middle_slice_idx])
                            except RuntimeError as e:
                                print(f"Error reading image at {scan_path}: {e}")
                                print("Skipping this image and continuing with the next one.")
                            except Exception as e:
                                print(f"Unexpected error processing image at {scan_path}: {e}")

        except Exception as e:
            print(f"Error processing scan ID {scan_id}: {e}")

    return list_middle_slices


def plot_from_scan_ids(path: str, scan_ids: list, sequences_wanted: list, output_folder: Optional[str] = None) -> None:
    """
    Plot middle slices from specified scans and sequences in a grid. Optionally save the plots to an output folder.

    Args:
        path (str): The root directory containing the patient data.
        scan_ids (list): A list of scan IDs to process.
        sequences_wanted (list): A list of sequences to consider.
        output_folder (Optional[str]): The directory to save the plots. If None, plots are not saved.
    """
    # Get the list of middle slices
    middle_slices = _get_list_middle_slices(path, scan_ids, sequences_wanted)

    if not middle_slices:
        print("No slices to plot.")
        return

    # Determine the number of rows and columns for the grid
    num_slices = len(middle_slices)
    num_cols = int(math.ceil(math.sqrt(num_slices)))
    num_rows = int(math.ceil(num_slices / num_cols))

    # Create the plot
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))
    fig.tight_layout()

    for i, slice in enumerate(middle_slices):
        ax = axes[i // num_cols, i % num_cols]
        ax.imshow(slice, cmap='gray')
        ax.axis('off')

    # Remove empty subplots
    for i in range(num_slices, num_rows * num_cols):
        fig.delaxes(axes[i // num_cols, i % num_cols])

    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    if output_folder:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_path = os.path.join(output_folder, 'middle_slices_grid.png')
        plt.savefig(output_path)
        print(f"Plots saved to {output_path}")
    else:
        plt.show()


plot_from_scan_ids(pathname, list_scan_ids, sequences_wanted)