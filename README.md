# Introduction

multi-QuCAD is a software tool for evaluating wait-time savings for Computer-Aided Triage and Notification (CADt) devices. Compared to its previous version [QuCAD](https://github.com/DIDSR/QuCAD), multi-QuCAD is also capable of simulating scenarios:
* with multiple disease conditions and multiple CADt devices in the same reading queue,
* including both a priroity workflow protovol (all AI-flagged cases are in the same priority class regardless of the targeted conditions of the AIs) and a hierarchical workflow protocol (AI-flagged cases by an AI is prioritized over AI-flagged cases by another AI), and
* assuming a non-preemptive priority discipline (i.e. radiologists are assumed to finish the lower-priority case in hand before reviewing the in-coming higher-priority case) in addition to the preemptive-resume priority schedule.

In the past decade, Artificial Intelligence (AI) algorithms have made promising impacts to transform health-care in all aspects. One application is to triage patients’ radiological medical images based on the algorithm’s binary outputs. Such AI-based prioritization software is known as computer-aided triage and notification (CADt). Their main benefit is to speed up radiological review of images with time-sensitive findings. However, as CADt devices become more common in clinical workflows, there is still a lack of quantitative methods to evaluate a device’s effectiveness in saving patients’ waiting times.

This software tool is developed to simulate clinical workflow of image review/interpretation. Included in this tool, we also provide a mathematical framework based on queueing theory to calculate the average waiting time per patient image before and after a CADt device is used. For more complex workflow model with multiple priority classes and radiologists, an approximation method known as the Recursive Dimensionality Reduction technique proposed by [Harchol-Balter et al (2005)](https://www.cs.cmu.edu/~harchol/Papers/questa.pdf) is applied. We define a performance metric to measure the device’s time-saving effectiveness. Simulated and theoretical average time-saving is comparable, and the simulation is used to provide confidence intervals of the performance metric we defined.

# Disclaimer
This software and documentation (the "Software") were developed at the Food and Drug Administration (FDA) by employees of the Federal Government in the course of their official duties. Pursuant to Title 17, Section 105 of the United States Code, this work is not subject to copyright protection and is in the public domain. Permission is hereby granted, free of charge, to any person obtaining a copy of the Software, to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, or sell copies of the Software or derivatives, and to permit persons to whom the Software is furnished to do so. FDA assumes no responsibility whatsoever for use by other parties of the Software, its source code, documentation or compiled executables, and makes no guarantees, expressed or implied, about its quality, reliability, or any other characteristic. Further, use of this code in no way implies endorsement by the FDA or confers any advantage in regulatory decisions. Although this software can be redistributed and/or modified freely, we ask that any derivative works bear some notice that they are derived from it, and any modified versions bear some notice that they have been modified.

# Software Requirements
This software package was developed using Python 3.9.4 with the following extra packages.
* numpy
* pandas
* scipy
* matplotlib
* statsmodels

`scripts/requirements.txt` contains a list of packages required to build a virtual enviornment to run this software.

# Software Usage
`scripts/run_sim.py` is the main script to run this software. It accepts user input values that specify clinical settings, CADt AI diagnostic performance, and software preferences. By default, outputs will be dumped in outputs/ automatically including plots and a pickled python dictionary that contains all simulation information. Please refer to  the `UserManual.pdf` and `scripts/README.md` for more information

```
$ cd /path/to/this/scripts/folder/
$ python run_sim.py --configFile ../inputs/config.dat
```

# Relevant Publications
* [Applying Queueing Theory to Evaluate Wait-Time-Savings of Triage Algorithms.](https://link.springer.com/article/10.1007/s11134-024-09927-w)
* [Wait-Time-Saving Analysis and Clinical Effectiveness of Computer-Aided Triage and Notification (CADt) Devices Based on Queueing Theory.](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12035/0000/Wait-time-saving-analysis-and-clinical-effectiveness-of-Computer-Aided/10.1117/12.2603184.short)
* [FDA Science Forum (2021)](https://www.fda.gov/media/148986/download)

# Citation for this tool
Yee Lam Elim Thompson, Michelle Mastrinanni, Rucha Deshpande, Jixin Audrey Zheng, Frank W. Samuelson. (2024) multi-QuCAD [Source Code] https://github.com/DIDSR/multi-QuCAD/.

# Contact
For any questions/suggestions/collaborations, please contact Elim Thompson either via this GitHub repo or via email (yeelamelim.thompson@fda.hhs.gov).
