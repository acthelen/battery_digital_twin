# battery_digital_twin

Questions regarding this repository should be directed to Adam Thelen (acthelen@iastate.edu).

$\ $

This repository contains the scripts and preprocessed data to recreate the figures and results presented in the paper:

**A Comprehensive Review of Digital Twin - Part 2: Roles of Uncertainty Quantification and Optimization, a Battery Digital Twin, and Perspectives**
  
Adam Thelen<sup>1</sup>, Xiaoge Zhang<sup>2</sup>, Olga Fink<sup>3</sup>, Yan Lu<sup>4</sup>, Sayan Ghosh<sup>5</sup>, Byeng D.
Youn<sup>6</sup>, Michael D. Todd<sup>7</sup>, Sankaran Mahadevan<sup>8</sup>, Chao Hu<sup>1*</sup> and Zhen Hu<sup>9*</sup>

1. Department of Mechanical Engineering, Iowa State University, Ames, 50011, IA, USA.
2. Department of Industrial and Systems Engineering, The Hong Kong Polytechnic University, Kowloon, Hong Kong.
3. Intelligent Maintenance and Operations Systems, Swiss Federal Institute of Technology Lausanne, Lausanne, 12309, NY, Switzerland.
4. Information Modeling and Testing Group, National Institute of Standards and Technology, Gaithersburg, 20877, MD, USA.
5. Probabilistic Design and Optimization group, GE Research, Niskayuna, 12309, NY, USA.
6. Department of Mechanical Engineering, Seoul National University, Gwanak-gu, 151-742, Seoul, Republic of Korea.
7. Department of Structural Engineering, University of California, San Diego, La Jolla, 92093, CA, USA.
8. Department of Civil and Environmental Engineering, Vanderbilt University, Nashville, 37235, TN, USA.
9. Department of Industrial and Manufacturing Systems Engineering, University of Michigan-Dearborn, Dearborn, 48128, MI, USA.

$\*$ Corresponding author(s). E-mail(s): chaohu@iastate.edu; zhennhu@umich.edu;

Contributing authors: acthelen@iastate.edu; xiaoge.zhang@polyu.edu.hk; olga.fink@epfl.ch; yan.lu@nist.gov; sayan.ghosh1@ge.com; bdyoun@snu.ac.kr; mdtodd@eng.ucsd.edu; sankaran.mahadevan@vanderbilt.edu


## Scripts Overview

The script `create_capacity_dataset_from_raw_files.py` reads the files in the folder `124 LFP Capacity Data` and compiles them into an easy to use dataset which is then saved as `capacity_dataset_124_lfp.pkl`. 

The particle filter code is presented in two different files. The first file, `plot_pf_projection_each_cycle_single_cell.py` is used to run the particle filter on a single cell. This scrip is best used for understanding how the particle filter works, as it shows how the particle filter prediction evolves over time as more capacity information is available. 

The second file, `run_pf_and_optimize_retirement_six_cells.py` runs the particle filter code for the six randomly selected cells shown in the paper, and optimizes the retirement of each cell individually. This script contains the multi-attribute utility theory (MAUT) algorithm described in the paper. This script is best used for understanding how the particle filter and the MAUT algorithm interact, and how the optimal retirement changes on a cell-by-cell basis. 


