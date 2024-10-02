#TODO: need to check if this works

import os
import tifffile
import torch
from torch.utils.data import Dataset, DataLoader

class LazyTIFFStackDataset(Dataset):
    def __init__(self, tiff_paths, transform=None):
        """
        Args:
            tiff_paths (list of str): List of paths to the TIFF stacks.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.tiff_paths = tiff_paths
        self.transform = transform

        # Calculate the cumulative timepoints across all stacks for indexing
        self.cumulative_timepoints = self.get_cumulative_timepoints()

    def get_cumulative_timepoints(self):
        """
        Calculates the cumulative number of timepoints in each TIFF stack.
        Returns:
            cumulative_timepoints (list): Cumulative sum of timepoints for each stack.
        """
        cumulative_timepoints = []
        total_timepoints = 0
        for path in self.tiff_paths:
            with tifffile.TiffFile(path) as tiff:
                timepoints = tiff.series[0].shape[0]  # Assumes shape is [timepoint, channel, y, x]
                total_timepoints += timepoints
                cumulative_timepoints.append(total_timepoints)
        return cumulative_timepoints

    def find_stack_index(self, global_idx):
        """
        Finds the index of the TIFF stack corresponding to a global index.
        Args:
            global_idx (int): The global index of the dataset.
        Returns:
            stack_idx (int): Index of the TIFF stack that contains the global_idx.
            local_idx (int): Local timepoint index within the found TIFF stack.
        """
        for stack_idx, cum_timepoint in enumerate(self.cumulative_timepoints):
            if global_idx < cum_timepoint:
                # Determine local index within the TIFF stack
                previous_timepoints = 0 if stack_idx == 0 else self.cumulative_timepoints[stack_idx - 1]
                local_idx = global_idx - previous_timepoints
                return stack_idx, local_idx
        raise IndexError("Index out of range for the dataset.")

    def __len__(self):
        # Return the total number of timepoints across all TIFF stacks
        return self.cumulative_timepoints[-1]

    def __getitem__(self, global_idx):
        """
        Get an individual sample from the dataset using lazy loading.
        Args:
            global_idx (int): Global index for accessing the dataset.
        Returns:
            sample (Tensor): The selected timepoint from the dataset.
        """
        # Find the corresponding TIFF stack and timepoint within that stack
        stack_idx, local_idx = self.find_stack_index(global_idx)

        # Load only the necessary TIFF stack (lazy loading)
        tiff_path = self.tiff_paths[stack_idx]
        with tifffile.TiffFile(tiff_path) as tiff:
            # Read the specific timepoint slice only (assumes shape [timepoint, channel, y, x])
            sample = tiff.asarray()[local_idx]  # Load only the timepoint `local_idx`

        # Convert to torch tensor
        sample = torch.tensor(sample, dtype=torch.float32)

        # Apply any transformations if specified
        if self.transform:
            sample = self.transform(sample)

        return sample

# Example usage with lazy loading

# List of paths to TIFF stacks
tiff_paths = [
    "/path/to/tiff_stack1.tif",
    "/path/to/tiff_stack2.tif",
    # Add more TIFF stack paths as needed
]

# Create lazy loading Dataset
lazy_dataset = LazyTIFFStackDataset(tiff_paths=tiff_paths)

# Create DataLoader
dataloader = DataLoader(lazy_dataset, batch_size=4, shuffle=True, num_workers=4)

# Iterate through the DataLoader
for batch in dataloader:
    print(batch.shape)  # Example output shape: [timePoint, channels, y, x]