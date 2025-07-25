#%% [markdown]
# Generate random raft indices for isolation.

#%%
from pathlib import Path
import csv
import random

#%%
dataDir_input = input('Isolation directory: ')
dataDir = Path(dataDir_input)

#%%
numArrs = 2
numRafts = 16

#%%
cenRafts = []
for r in range(2, 21):
    for c in range(2, 21):
        cenRafts.append((r * 23) + (c + 1))

#%%
table = [['x' for _ in range(12)] for _ in range(8)]

for arr in range(numArrs):
    random_numbers = random.sample(cenRafts, numRafts)  # Ensures unique selections
    offset = int((12 / numArrs) * arr)

    index = 0
    for col in range(offset, 12):
        for row in range(8):
            if index < numRafts:
                table[row][col] = random_numbers[index]
                index += 1

print(table)

#%%
file_path = dataDir / 'Raft Isolation Map.csv'
with open(file_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(table)

print(f'CSV file saved as {file_path}')