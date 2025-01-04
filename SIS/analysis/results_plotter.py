
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as colors

input_csv = "test_margins.csv"
# # input_csv = "test_margins_2.csv"

m_vals = 100*2**np.array([1, 2, 3, 4, 5, 6]) # 200, 400, 800, 1600, 3200, 6400
n_vals = 10*2**np.array([1, 2, 3, 4, 5, 6]) # 20, 40, 80, 160, 320, 640

# input_csv = "real_degrees.csv"

# m_vals = 100*2**np.array([1, 2, 3, 4, 5, 6,7]) # 200, 400, 800, 1600, 3200, 6400
# n_vals = 10*2**np.array([1, 2, 3, 4, 5, 6,7]) # 20, 40, 80, 160, 320, 640

results_df = pd.read_csv(input_csv)
print(results_df)

m_ticks = [] # Centers of the bins in m space
m_bin_upper_bounds = [] # (exclusive) Upper bounds of the m bins
m_tick_labels = []
n_ticks = [] # Centers of the bins in n space
n_bin_upper_bounds = [] # (exclusive) Upper bounds of the n bins

m_bin_upper_bounds = m_vals + 1
n_bin_upper_bounds = n_vals + 1

error_examples = []
for i in range(len(m_vals)):
    error_examples.append([])
    for j in range(len(n_vals)):
        error_examples[i].append([])


for row_tuple in results_df.iterrows():
    filename = row_tuple[1][results_df.columns.get_loc('filename')]
    m = row_tuple[1][results_df.columns.get_loc('m')]
    n = row_tuple[1][results_df.columns.get_loc('n')]
    value = row_tuple[1][results_df.columns.get_loc('value')]
    error = row_tuple[1][results_df.columns.get_loc('error')]
    estimate = row_tuple[1][results_df.columns.get_loc('estimate')]
    # estimate = row_tuple[1][results_df.columns.get_loc('estimate_3')]


    m_ind = -1
    n_ind = -1
    for s in range(len(m_bin_upper_bounds)):
        if m < m_bin_upper_bounds[s]:
            m_ind = s
            break
    for s in range(len(n_bin_upper_bounds)):
        if n < n_bin_upper_bounds[s]:
            n_ind = s
            break
    print(m_ind,n_ind,filename)
    error_examples[m_ind][n_ind].append((estimate - value)/value)
    print(value,row_tuple[1][results_df.columns.get_loc('estimate')],row_tuple[1][results_df.columns.get_loc('estimate_3')])

# print(error_examples)
error_means = np.zeros((len(m_vals),len(n_vals)))
error_rmses = np.zeros((len(m_vals),len(n_vals)))
for i in range(len(m_vals)):
    for j in range(len(n_vals)):
        if len(error_examples[i][j]) > 0:
            error_means[i][j] = np.mean(error_examples[i][j])
            error_rmses[i][j] = np.sqrt(np.mean(np.array(error_examples[i][j])**2))
        else:
            error_means[i][j] = -1
            error_rmses[i][j] = -1

# print(error_means)
# Grid showing the means and stds on a log scale
fig, ax = plt.subplots()
cmap = cm.get_cmap("plasma").copy()  # Can be any colormap that you want after the cm
cmap.set_bad(color='white')
i = ax.imshow(error_rmses.T,norm=colors.LogNorm(vmin=10**(-6), vmax=10**(-1)),
                    cmap=cmap)
fig.colorbar(i)
ax.set_xlabel('m')
ax.set_xticks(range(len(m_vals)))
ax.set_xticklabels(m_vals)
ax.set_ylabel('n')
ax.set_yticks(range(len(n_vals)))
ax.set_yticklabels(n_vals)
ax.invert_yaxis()
ax.set_title('Relative RMSE in log Omega Estimate')
plt.savefig(f'{input_csv[:-4]}_rmse_error.png')



# symlog for the mean
fig, ax = plt.subplots()
# Use a color map that is red for negative, white for zero, and blue for positive
cmap = cm.get_cmap("coolwarm").copy()  # Can be any colormap that you want after the cm
i = ax.imshow(error_means.T,norm=colors.SymLogNorm(linthresh=10**(-6),linscale=10**(-6),vmin=-10**(-1), vmax=10**(-1)),
                    cmap=cmap)
fig.colorbar(i)
ax.set_xlabel('m')
ax.set_xticks(range(len(m_vals)))
ax.set_xticklabels(m_vals)
ax.set_ylabel('n')
ax.set_yticks(range(len(n_vals)))
ax.set_yticklabels(n_vals)
ax.invert_yaxis()
ax.set_title('Mean Relative Error in log Omega Estimate')
plt.savefig(f'{input_csv[:-4]}_mean_error.png')

# extras = ['SIS','logOmega']
# for e in extras:
#   array = []
#   for i in range(len(mn_ticks)):
#     array.append([])
#     for j in range(len(N_ticks)):
#       array[i].append([])
#   error_examples[e] = array
# for k in estimates:
#   array = []
#   for i in range(len(mn_ticks)):
#     array.append([])
#     for j in range(len(N_ticks)):
#       array[i].append([])
#   error_examples[k] = array

# for row_tuple in results_df.iterrows():
#   N = row_tuple[1][results_df.columns.get_loc('N')]
#   m = row_tuple[1][results_df.columns.get_loc('m')]
#   n = row_tuple[1][results_df.columns.get_loc('n')]
#   value = row_tuple[1][results_df.columns.get_loc('value')]
#   error = row_tuple[1][results_df.columns.get_loc('error')]
#   N_ind = -1
#   mn_ind = -1
#   for s in range(len(N_bin_upper_bounds)):
#     if N < N_bin_upper_bounds[s]:
#       N_ind = s
#       break
#   for s in range(len(mn_bin_upper_bounds)):
#     if m*n < mn_bin_upper_bounds[s]:
#       mn_ind = s
#       break
  
  

#   error_examples['logOmega'][mn_ind][N_ind].append(value)
#   error_examples['SIS'][mn_ind][N_ind].append(abs(error/value))
#   for est_name in estimates:
#     estimate = row_tuple[1][results_df.columns.get_loc(est_name)]
#     if estimate != -1:
#       error_examples[est_name][mn_ind][N_ind].append(abs((estimate-value)/value))
#       # Average absolute relative error (I suppose this could instead be the RMS error)

# meanSISErrors = []
# for i in range(len(error_examples['SIS'])):
#   meanSISErrors.append([])
#   for j in range(len(error_examples['SIS'][i])):
#     meanSISErrors[i].append(np.mean(error_examples['SIS'][i][j]))
# SIS_error_threshold = 10**(-3)

# error_arrays_square = {}
# for est_name, example_array in error_examples.items():
#   array = -1*np.ones((len(mn_ticks),len(N_ticks)))
#   for i in range(len(example_array)):
#     for j in range(len(example_array[i])):
#       if len(example_array[i][j]) > 0 and meanSISErrors[i][j] < SIS_error_threshold:
#         array[i][j] = max(np.mean(example_array[i][j]),10**(-10))
#   error_arrays_square[est_name] = np.array(array)

# # Load Size Trial
# estimatesFilename = "SIS-logOmega-Estimates-Size.csv"
# # estimatesFilename = "SIS-logOmega-Estimates.txt"
# results_df = pd.read_csv(estimatesFilename)
# default_headers = ['m','n','N','name','value','error']
# estimates = [i for i in results_df.keys() if i not in default_headers]
# print('Estimates:',estimates)

# m_ticks = [] # Centers of the bins in mn space
# m_bin_upper_bounds = [] # (exclusive) Upper bounds of the n bins
# m_tick_labels = []

# n_ticks = [] # Centers of the bins in mn space
# n_bin_upper_bounds = [] # (exclusive) Upper bounds of the n bins
# n_tick_labels = []


# for s in range(2,11):
#   m_ticks.append(pow(2,s))
#   m_bin_upper_bounds.append(1.4*pow(2,s))
#   m_tick_labels.append(str(pow(2,s)))

# for s in range(2,11):
#   n_ticks.append(pow(2,s))
#   n_bin_upper_bounds.append(1.4*pow(2,s))
#   n_tick_labels.append(str(pow(2,s)))

# print(m_ticks,n_ticks)
# error_examples = {}

# extras = ['SIS','logOmega']
# for e in extras:
#   array = []
#   for i in range(len(m_ticks)):
#     array.append([])
#     for j in range(len(n_ticks)):
#       array[i].append([])
#   error_examples[e] = array
# for k in estimates:
#   array = []
#   for i in range(len(m_ticks)):
#     array.append([])
#     for j in range(len(n_ticks)):
#       array[i].append([])
#   error_examples[k] = array

# for row_tuple in results_df.iterrows():
#   N = row_tuple[1][results_df.columns.get_loc('N')]
#   m = row_tuple[1][results_df.columns.get_loc('m')]
#   n = row_tuple[1][results_df.columns.get_loc('n')]
#   value = row_tuple[1][results_df.columns.get_loc('value')]
#   error = row_tuple[1][results_df.columns.get_loc('error')]
#   m_ind = -1
#   n_ind = -1
#   for s in range(len(m_bin_upper_bounds)):
#     if m < m_bin_upper_bounds[s]:
#       m_ind = s
#       break
#   for s in range(len(n_bin_upper_bounds)):
#     if n < n_bin_upper_bounds[s]:
#       n_ind = s
#       break
  
#   error_examples['logOmega'][m_ind][n_ind].append(value)
#   error_examples['SIS'][m_ind][n_ind].append(abs(error/value))
#   for est_name in estimates:
#     estimate = row_tuple[1][results_df.columns.get_loc(est_name)]
#     if estimate != -1:
#       error_examples[est_name][m_ind][n_ind].append(abs((estimate-value)/value))

# meanSISErrors = []
# for i in range(len(error_examples['SIS'])):
#   meanSISErrors.append([])
#   for j in range(len(error_examples['SIS'][i])):
#     meanSISErrors[i].append(np.mean(error_examples['SIS'][i][j]))
# SIS_error_threshold = 10**(-3)

# error_arrays = {}
# for est_name, example_array in error_examples.items():
#   array = -1*np.ones((len(m_ticks),len(n_ticks)))
#   for i in range(len(example_array)):
#     for j in range(len(example_array[i])):
#       if len(example_array[i][j]) > 0 and meanSISErrors[i][j] < SIS_error_threshold:
#         array[i][j] = max(np.mean(example_array[i][j]),10**(-10))
#   error_arrays[est_name] = np.array(array)

# minError = 10**(-6)
# maxError = 10**(-1)

# for est_name, error_array in error_arrays_square.items():
#   fig, ax = plt.subplots()
#   cmap = cm.get_cmap("plasma").copy()  # Can be any colormap that you want after the cm
#   cmap.set_bad(color='white')
#   if est_name == 'logOmega':
#     i = ax.imshow(error_array,norm=colors.LogNorm(vmin=1, vmax=100000),
#                       cmap=cmap)
#     fig.colorbar(i)
#   else:
#     i = ax.imshow(error_array,norm=colors.LogNorm(vmin=minError, vmax=maxError),
#                       cmap=cmap)
#     fig.colorbar(i)
#   ax.set_xlabel('N')
#   ax.set_xticks(range(len(N_ticks)))
#   ax.set_xticklabels(N_ticks)
#   ax.set_ylabel('m x n')
#   ax.set_yticks(range(len(mn_ticks)))
#   ax.set_yticklabels(mn_tick_labels)
#   ax.invert_yaxis()
#   ax.set_title(r'Fractional Error in $\log \Omega^{'+est_name+r'}(\mathbf{r},\mathbf{c})$')
#   if est_name != "SIS":
#     for i in range(len(error_array)): # If not SIS, draw an asterisk over the regions where the
#       for j in range(len(error_array[0])): # benchmark error is within 10x the SIS error
#         if error_array[i][j] != -1 and error_array[i][j] < 5*error_arrays_square['SIS'][i][j]:
#           text = ax.text(j, i, "*",
#                         ha="center",size=20, va="center", color="gray")
#   plt.savefig(join(graphics_folder,est_name+'.pdf'))

#   fig = plt.figure(figsize=(9., 6.))
#   grid = ImageGrid(fig, 111,  # similar to subplot(111)
#                   nrows_ncols=(1, 2),  # creates 2x2 grid of axes
#                   axes_pad=0.4,  # pad between axes in inch.
#                   #cbar_pad=0.1,
#                   )

#   cmap = cm.get_cmap("plasma").copy()  # Can be any colormap that you want after the cm
#   cmap.set_bad(color='white')
#   if est_name == 'logOmega':
#     i = grid[0].imshow(error_array,norm=colors.LogNorm(vmin=1, vmax=100000),
#                       cmap=cmap)
#     fig.colorbar(i)
#   else:
#     i = grid[0].imshow(error_array,norm=colors.LogNorm(vmin=minError, vmax=maxError),
#                       cmap=cmap)
#     fig.colorbar(i)
#   grid[0].set_xlabel('N')
#   grid[0].set_xticks(range(len(N_ticks)))
#   grid[0].set_xticklabels(N_ticks)
#   grid[0].set_ylabel('m x n')
#   grid[0].set_yticks(range(len(mn_ticks)))
#   grid[0].set_yticklabels(mn_tick_labels)
#   grid[0].invert_yaxis()
#   if est_name == 'logOmega':
#     i = grid[1].imshow(error_arrays[est_name],norm=colors.LogNorm(vmin=1, vmax=100000),
#                       cmap=cmap)
#   else:
#     i = grid[1].imshow(error_arrays[est_name],norm=colors.LogNorm(vmin=minError, vmax=maxError),
#                       cmap=cmap)
#     fig.colorbar(i)
#   grid[1].set_xlabel('n')
#   grid[1].set_xticks(range(len(n_ticks)))
#   grid[1].set_xticklabels(n_tick_labels)
#   grid[1].set_ylabel('m')
#   grid[1].set_yticks(range(len(m_ticks)))
#   grid[1].set_yticklabels(m_tick_labels)
#   grid[1].invert_yaxis()
#   plt.savefig(join(graphics_folder,est_name+'Combined.pdf'))

# fig = plt.figure(figsize=(9., 6.))
# grid = ImageGrid(fig, 111,  # similar to subplot(111)
#                  nrows_ncols=(2, 3),  # creates 2x2 grid of axes
#                  axes_pad=0.4,  # pad between axes in inch.
#                  #cbar_pad=0.1,
#                  )

# grid_names = ["EC", "GM", "DE", "BBK","GMW","GC"]
# grid_arrays = []
# for name in grid_names:
#   grid_arrays.append(error_arrays_square[name])

# for ax, im, est_name in zip(grid, grid_arrays, grid_names):
#   # Iterating over the grid returns the Axes.
#   ax.imshow(im)
#   imshown = ax.imshow(im,norm=colors.LogNorm(vmin=minError, vmax=maxError),
#                     cmap=cmap)
#   ax.set_xlabel('N')
#   ax.set_xticks(range(len(N_ticks)))
#   ax.set_xticklabels(N_ticks, rotation = 45)
#   ax.set_ylabel('m x n')
#   ax.set_yticks(range(len(mn_ticks)))
#   ax.set_yticklabels(mn_tick_labels)
#   ax.invert_yaxis()
#   ax.set_title(est_name)
#   if est_name != "SIS":
#     for i in range(len(im)): # If not SIS, draw an asterisk over the regions where the
#       for j in range(len(im[0])): # benchmark error is within 10x the SIS error
#         if im[i][j] != -1 and im[i][j] < 5*error_arrays_square['SIS'][i][j]:
#           text = ax.text(j, i, "*",
#                         ha="center",size=15, va="center", color="gray")

# #fig.colorbar(imshown,ax=ax,location='right')
# #grid.cbar_axes[0].colorbar(imshown)


# plt.suptitle(r'Fractional Error in Linear-time Estimates of $\log \Omega(\mathbf{r},\mathbf{c})$')
# # plt.tight_layout()
# plt.savefig(join(graphics_folder,'Grid.pdf'))

# ##################

# fig = plt.figure(figsize=(6., 4.))
# grid = ImageGrid(fig, 111,  # similar to subplot(111)
#                  nrows_ncols=(1, 2),  # creates 2x2 grid of axes
#                  axes_pad=0.4,  # pad between axes in inch.
#                  #cbar_pad=0.1,
#                  )

# grid_names = ["ME-G","ME-E"]
# grid_arrays = []
# for name in grid_names:
#   grid_arrays.append(error_arrays_square[name])

# for ax, im, est_name in zip(grid, grid_arrays, grid_names):
#   # Iterating over the grid returns the Axes.
#   ax.imshow(im)
#   imshown = ax.imshow(im,norm=colors.LogNorm(vmin=minError, vmax=maxError),
#                     cmap=cmap)
#   ax.set_xlabel('N')
#   ax.set_xticks(range(len(N_ticks)))
#   ax.set_xticklabels(N_ticks, rotation = 45)
#   ax.set_ylabel('m x n')
#   ax.set_yticks(range(len(mn_ticks)))
#   ax.set_yticklabels(mn_tick_labels)
#   ax.invert_yaxis()
#   ax.set_title(est_name)
#   if est_name != "SIS":
#     for i in range(len(im)): # If not SIS, draw an asterisk over the regions where the
#       for j in range(len(im[0])): # benchmark error is within 10x the SIS error
#         if im[i][j] != -1 and im[i][j] < 5*error_arrays_square['SIS'][i][j]:
#           text = ax.text(j, i, "*",
#                         ha="center",size=15, va="center", color="gray")

# #fig.colorbar(imshown,ax=ax,location='right')
# #grid.cbar_axes[0].colorbar(imshown)

# plt.suptitle(r'Fractional Error in Maximum-Entropy Estimates of $\log \Omega(\mathbf{r},\mathbf{c})$')
# # plt.tight_layout()
# plt.savefig(join(graphics_folder,'GridME.pdf'))
