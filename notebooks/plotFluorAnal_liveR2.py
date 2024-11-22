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

df_fluors = pd.DataFrame()

for array in arrays:
    for idx, ploid in enumerate(ploids):
        df_fluor = pd.read_csv(f'{ploid}_array{array}_FluorAnalysis_CalcHoechst_Raw.csv')
        df_fluor_filter = df_fluor[(df_fluor['Missing'] == 0) &
                                   (df_fluor['Empty'] == 0) &
                                   (df_fluor['Border'] == 0)]
        df_fluor_filter['Array'] = [array] * df_fluor_filter.shape[0]
        df_fluor_filter['Ploidy'] = [ploid_labs[idx]] * df_fluor_filter.shape[0]
        df_fluor_filter['FArea_Hull'] = [(fArea * pxSize**2)
                                         for fArea in df_fluor_filter['FArea_Hull'].tolist()]

        df_fluor_melt = df_fluor_filter.melt(id_vars=['Array', 'Ploidy', 'Raft'],
                                         value_vars=['NucDen', 'FArea_Hull']
                                         )

        df_fluors = pd.concat([df_fluors, df_fluor_melt],
                               axis=0,
                               ignore_index=True)

#%%
arrs_col2 = []
ploids_col2 = []
counts_col2 = []
fts_col2 = []
means_col2 = []
stdevs_col2 = []

for array in arrays:
    for ploid in ploid_labs:
        for ft in ['NucDen', 'FArea_Hull']:
            df_sub = df_fluors[(df_fluors['Array'] == array) &
                               (df_fluors['Ploidy'] == ploid) &
                               (df_fluors['variable'] == ft)]
            count = df_sub.shape[0]
            mean = np.mean(df_sub['value'])
            stdev = np.std(df_sub['value'])

            arrs_col2.append(array)
            ploids_col2.append(ploid)
            counts_col2.append(count)
            fts_col2.append(ft)
            means_col2.append(mean)
            stdevs_col2.append(stdev)

df_means = pd.DataFrame({'Array': arrs_col2,
                         'Ploidy': ploids_col2,
                         'Count': counts_col2,
                         'Feat': fts_col2,
                         'Mean': means_col2,
                         'Stdev': stdevs_col2})

#%%
df_means_eu = df_means[df_means['Ploidy'] == 'Euploid']
df_means_an = df_means[df_means['Ploidy'] == 'Aneuploid']
df_means_list = [df_means_eu, df_means_an]

ttest_nucDen = stats.ttest_ind(a=df_means_eu[df_means_eu['Feat'] == 'NucDen']['Mean'],
                               b=df_means_an[df_means_an['Feat'] == 'NucDen']['Mean'],
                               equal_var=True)
print(f'NucDen: {ttest_nucDen}')

ttest_hull = stats.ttest_ind(a=df_means_eu[df_means_eu['Feat'] == 'FArea_Hull']['Mean'],
                               b=df_means_an[df_means_an['Feat'] == 'FArea_Hull']['Mean'],
                               equal_var=True)
print(f'Hull: {ttest_hull}')

#%%
ploids_col3 = []
feats_col3 = []
means_col3 = []
sems_col3 = []

for idx, ploid in enumerate(ploid_labs):
    for ft in ['NucDen', 'FArea_Hull']:
        df_means_ploid = df_means_list[idx]
        df_means_ft = df_means_ploid[df_means_ploid['Feat'] == ft]

        print(df_means_ft['Mean'].tolist())
        mean = np.mean(df_means_ft['Mean'].tolist())
        sem = np.std(df_means_ft['Mean'].tolist()) / np.sqrt(len(df_means_ft['Mean'].tolist()))

        # means = np.array(df_means_ft['Mean'].tolist())
        # sampSizes = np.array(df_means_ft['Count'].tolist())
        # stdevs = np.array(df_means_ft['Stdev'].tolist())
        #
        # weightMean = np.sum(means * sampSizes) / np.sum(sampSizes)
        # weightVar = np.sum(sampSizes * (stdevs**2 + (means - weightMean)**2)) / np.sum(sampSizes)
        # weightStdev = np.sqrt(weightVar)
        # weightSE = weightStdev / np.sqrt(np.sum(sampSizes))

        ploids_col3.append(ploid)
        feats_col3.append(ft)
        means_col3.append(mean)
        sems_col3.append(sem)

df_means2 = pd.DataFrame({'Ploidy': ploids_col3,
                          'Feat': feats_col3,
                          'Mean': means_col3,
                          'SEM': sems_col3})

#%%
df_means.to_csv((dataDir.parent / 'Eu & Aneu Gastruloid Nuclei Density_Mean.csv'),
                index=False)
df_means2.to_csv((dataDir.parent / 'Eu & Aneu Gastruloid Nuclei Density_Mean2.csv'),
                index=False)

#%%
df_fluors_nucDen = df_fluors[df_fluors['variable'] == 'NucDen']
df_fluors_nucDen.reset_index(drop=True, inplace=True)
df_means_nucDen = df_means[df_means['Feat'] == 'NucDen']
df_means_nucDen.reset_index(drop=True, inplace=True)
df_means2_nucDen = df_means2[df_means2['Feat'] == 'NucDen']
df_means2_nucDen.reset_index(drop=True, inplace=True)

plt.rcdefaults()
sns.set_theme(style='white', font_scale=1.25)
plt.rcParams['ytick.left'] = True
fig, axes = plt.subplots(1, 1, figsize=(5, 8))

sns.stripplot(x='Ploidy',
              y='value',
              data = df_fluors_nucDen,
              hue='Array',
              size=3,
              alpha=0.25,
              palette='colorblind',
              jitter=0.4,
              legend=False,
              zorder=1)
sns.swarmplot(x='Ploidy',
              y='Mean',
              data=df_means_nucDen,
              hue='Array',
              size=15,
              palette='colorblind',
              edgecolor='k',
              linewidth=2,
              legend=False,
              zorder=2)
axes.errorbar(x=df_means2_nucDen['Ploidy'],
             y=df_means2_nucDen['Mean'],
             yerr=df_means2_nucDen['SEM'],
             fmt='none',
             color='black',
             capsize=10,
             capthick=3,
             marker='_',
             markersize=40,
             markeredgewidth=3,
             linewidth=3,
             zorder=3)
for i, row in df_means2_nucDen.iterrows():
    axes.hlines(y=row['Mean'],
               xmin= (i - 0.3),
               xmax= (i + 0.3),
               colors='black',
               linewidth=3,
               zorder=3)

plt.plot([0, 0, 1, 1],
         [max(df_fluors_nucDen['value'])+0.01,
          max(df_fluors_nucDen['value'])+0.02,
          max(df_fluors_nucDen['value'])+0.02,
          max(df_fluors_nucDen['value'])+0.01],
         linewidth=2,
         color='k')
plt.text((0+1)*.5,
         max(df_fluors_nucDen['value'])+0.005,
         "***",
         ha='center',
         va='bottom',
         color='k',
         fontsize=20)

axes.set_xlabel('')
axes.set_xticks([0, 1], ploid_labs, fontweight='bold')
axes.set_xlim(-0.5, 1.5)
axes.set_ylabel('Nuclei Density', fontweight='bold')
axes.set_ylim(-0.01, 1.06)

plt.tight_layout()
plt.show()

#%%
df_fluors_hull = df_fluors[df_fluors['variable'] == 'FArea_Hull']
df_fluors_hull.reset_index(drop=True, inplace=True)
df_means_hull = df_means[df_means['Feat'] == 'FArea_Hull']
df_means_hull.reset_index(drop=True, inplace=True)
df_means2_hull = df_means2[df_means2['Feat'] == 'FArea_Hull']
df_means2_hull.reset_index(drop=True, inplace=True)

plt.rcdefaults()
sns.set_theme(style='white', font_scale=1.25)
plt.rcParams['ytick.left'] = True
fig, axes = plt.subplots(1, 1, figsize=(4, 4))

sns.stripplot(x='Ploidy',
              y='value',
              data = df_fluors_hull,
              hue='Array',
              size=3,
              alpha=0.25,
              palette='colorblind',
              jitter=0.4,
              legend=False,
              zorder=1)
sns.swarmplot(x='Ploidy',
              y='Mean',
              data=df_means_hull,
              hue='Array',
              size=15,
              palette='colorblind',
              edgecolor='k',
              linewidth=2,
              legend=False,
              zorder=2)
axes.errorbar(x=df_means2_hull['Ploidy'],
             y=df_means2_hull['Mean'],
             yerr=df_means2_hull['SEM'],
             fmt='none',
             color='black',
             capsize=10,
             capthick=3,
             marker='_',
             markersize=40,
             markeredgewidth=3,
             linewidth=3,
             zorder=3)
for i, row in df_means2_hull.iterrows():
    axes.hlines(y=row['Mean'],
               xmin= (i - 0.3),
               xmax= (i + 0.3),
               colors='black',
               linewidth=3,
               zorder=3)

plt.plot([0, 0, 1, 1],
         [max(df_fluors_hull['value'])+1e4,
          max(df_fluors_hull['value'])+1.5e4,
          max(df_fluors_hull['value'])+1.5e4,
          max(df_fluors_hull['value'])+1e4],
         linewidth=2,
         color='k')
plt.text((0+1)*.5,
         max(df_fluors_hull['value'])+1.5e4,
         "n.s.",
         ha='center',
         va='bottom',
         color='k'
         )

axes.set_xlabel('')
axes.set_xticks([0, 1], ploid_labs, fontweight='bold')
axes.set_xlim(-0.5, 1.5)
axes.set_ylabel(r'Fluorescence Area ($\mathregular{10^5}$ $\mathregular{μm^2}$)',
                fontweight='bold')
axes.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
axes.yaxis.get_offset_text().set_visible(False)
axes.set_ylim(min(df_fluors_hull['value']) * 0.85, max(df_fluors_hull['value']) * 1.12)

plt.tight_layout()
plt.show()
