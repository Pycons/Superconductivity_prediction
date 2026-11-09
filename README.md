The folder 'data' contains the dataset downloaded from the GitHub repository of the reference paper. The data are split into two files:
- 'success.csv': superconducting compounds;
- 'fail.csv': non-superconducting compounds, which have a critical temperature of zero.
Both files have the same four columns:
1. *(unnamed)*: row index within the file, starting from 0;
2. 'Unnamed: 0': index of the compound in the original, complete dataset before it was split into the two files;
3. 'Compound': chemical formula of the material;
4. 'Tc': measured critical temperature, in Kelvin.

The file 'make_dataset.py' is used to

The file 'plot dataset.py' is used to create some plots that are usefull to visualize the dataset and understand the characteristics of the material it contains. Precisely it plots the distribution of the critical temperatures, the element distribution and the most frequent elements. These plots are saved in the folder 'data_plots' in the folder 'plots'. 

The file 'make_model.py' is the core of the project, it defines the class TwoHeadNet, that inherits
