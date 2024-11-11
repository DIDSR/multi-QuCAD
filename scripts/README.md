## Scripts folder

This scripts folder contains all the python scripts to take user input values, run simulations, calculate predictions, and generate plots. Please refer to `tools/` sub folder for the details on how simulation and calculation is done. This folder contains two main files.

`requirements.txt` is the text file listed out all the packages needed for this software. Please refer to the `UserManual.pdf` for more information on how to use this file to set up a virtual environment.

`run_sim.py` is the main python script to simulate radiology reading workflow at a specific clinical setting with a CADt diagnostic performance. This simulation software handles a simplified scenario with 1 AI that is trained to identify 1 disease condition from 1 modality and anatomy. Patients in the reading queue either have the disease condition or not.

User can run it via an input file `config.dat`.

```
$ python run_sim.py --configFile ../inputs/config.dat
```

`show_results.py` contains functions to display theoretical and simulation results after a run is completed.

For more information on input parameters, please refer to the `UserManual.pdf`.