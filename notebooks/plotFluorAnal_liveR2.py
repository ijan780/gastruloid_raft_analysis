#%%
import os
from pathlib import Path
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import scipy.stats as stats

#%%
dataDir =  Path(r'C:\Users\MVX\OneDrive - UW\Allbritton Lab\Manuscripts\Manuscript 1 - Technical\Data\Eu & Aneu Gastruloid Fluor Analysis')
os.chdir(dataDir)

arrays = [_ for _ in range(1, 7)]
ploids= ['EU', 'AN']
ploid_labs = ['Euploid', 'Aneuploid']
pxSize = 1.3163  # um/px

df_nucDens = pd.DataFrame()

for array in arrays:
    for idx, ploid in enumerate(ploids):
        df_fluor = pd.read_csv(f'{ploid}_array{array}_FluorAnalysis_CalcHoechst.csv')
        df_fluor_filter = df_fluor[(df_fluor['Missing'] == 0) &
                                   (df_fluor['Empty'] == 0) &
                                   (df_fluor['Border'] == 0)]
        df_fluor_filter['Array'] = [array] * df_fluor_filter.shape[0]
        df_fluor_filter['Ploidy'] = [ploid_labs[idx]] * df_fluor_filter.shape[0]

        df_nucDen = df_fluor_filter.melt(id_vars=['Array', 'Ploidy', 'Raft'],
                                         value_vars='NucDen'
                                         )

        df_nucDens = pd.concat([df_nucDens, df_nucDen],
                               axis=0,
                               ignore_index=True)

#%%
arrs_col2 = []
ploids_col2 = []
counts_col2 = []
means_col2 = []
stdevs_col2 = []

for array in arrays:
    for ploid in ploid_labs:
        df_sub = df_nucDens[(df_nucDens['Array'] == array) &
                            (df_nucDens['Ploidy'] == ploid)]
        count = df_sub.shape[0]
        mean = np.mean(df_sub['value'])
        stdev = np.std(df_sub['value'])

        arrs_col2.append(array)
        ploids_col2.append(ploid)
        counts_col2.append(count)
        means_col2.append(mean)
        stdevs_col2.append(stdev)

df_means = pd.DataFrame({'Array': arrs_col2,
                         'Ploidy': ploids_col2,
                         'Count': counts_col2,
                         'Mean': means_col2,
                         'Stdev': stdevs_col2})

#%%
df_means_eu = df_means[df_means['Ploidy'] == 'Euploid']
df_means_an = df_means[df_means['Ploidy'] == 'Aneuploid']
df_means_list = [df_means_eu, df_means_an]

ttest_nucDen = stats.ttest_ind(a=df_means_eu['Mean'],
                               b=df_means_an['Mean'],
                               equal_var=True)
print(ttest_nucDen)

ploids_col3 = []
weightMeans_col3 = []
weightSEs_col3 = []

for idx, ploid in enumerate(ploid_labs):
    df_means_ploid = df_means_list[idx]

    means = np.array(df_means_ploid['Mean'].tolist())
    sampSizes = np.array(df_means_ploid['Count'].tolist())
    stdevs = np.array(df_means_ploid['Stdev'].tolist())

    weightMean = np.sum(means * sampSizes) / np.sum(sampSizes)
    weightVar = np.sum(sampSizes * (stdevs**2 + (means - weightMean)**2)) / np.sum(sampSizes)
    weightStdev = np.sqrt(weightVar)
    weightSE = weightStdev / np.sqrt(np.sum(sampSizes))

    ploids_col3.append(ploid)
    weightMeans_col3.append(weightMean)
    weightSEs_col3.append(weightSE)

df_weight = pd.DataFrame({'Ploidy': ploids_col3,
                          'WeightMean': weightMeans_col3,
                          'WeightSE': weightSEs_col3})

#%%
df_means.to_csv((dataDir.parent / 'Eu & Aneu Gastruloid Nuclei Density_Mean.csv'),
                index=False)
df_weight.to_csv((dataDir.parent / 'Eu & Aneu Gastruloid Nuclei Density_Weighted Mean.csv'),
                index=False)

#%%
plt.rcdefaults()
sns.set_theme(style='white', font_scale=1.25)
plt.rcParams['ytick.left'] = True
fig, axes = plt.subplots(1, 1, figsize=(4, 8))

sns.stripplot(x='Ploidy',
              y='value',
              data = df_nucDens,
              hue='Array',
              size=3,
              alpha=0.25,
              palette='colorblind',
              jitter=0.4,
              legend=False,
              zorder=1)
sns.swarmplot(x='Ploidy',
              y='Mean',
              data=df_means,
              hue='Array',
              size=15,
              palette='colorblind',
              edgecolor='k',
              linewidth=1,
              legend=False,
              zorder=2)
axes.errorbar(x=df_weight['Ploidy'],
             y=df_weight['WeightMean'],
             yerr=df_weight['WeightSE'],
             fmt='none',
             color='black',
             capsize=10,
             capthick=3,
             marker='_',
             markersize=40,
             markeredgewidth=3,
             linewidth=3,
             zorder=3)
for i, row in df_weight.iterrows():
    axes.hlines(y=row['WeightMean'],
               xmin= (i - 0.3),
               xmax= (i + 0.3),
               colors='black',
               linewidth=3,
               zorder=3)

plt.plot([0, 0, 1, 1],
         [max(df_means['Mean'])+0.05,
          max(df_means['Mean'])+0.06,
          max(df_means['Mean'])+0.06,
          max(df_means['Mean'])+0.05],
         linewidth=2,
         color='k')
plt.text((0+1)*.5,
         max(df_means['Mean'])+0.05,
         "***",
         ha='center',
         va='bottom',
         color='k',
         fontsize=20)

axes.set_xlabel('')
axes.set_xticks([0, 1], ploid_labs, fontweight='bold')
axes.set_xlim(-0.5, 1.5)
axes.set_ylabel('Nuclei Density', fontweight='bold')
axes.set_ylim(-0.01, 1.07)

# plt.plot([0, 0, 1, 1], [max(means)+0.075, max(means)+0.1, max(means)+0.1, max(means)+0.075], linewidth=2, color='k')
# plt.text((0+1)*.5, max(means)+0.075, "***", ha='center', va='bottom', color='k', fontsize=20)

plt.tight_layout()
plt.show()