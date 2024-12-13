##
## By Elim Thompson (11/27/2020)
##
## This script includes functions that plot image workflow and various
## distributions such as number of patient iamges in queue and their wait
## time for different priority classes.
###########################################################################

################################
## Import packages
################################ 
import numpy, matplotlib

matplotlib.use ('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import warnings
warnings.filterwarnings("ignore")

from . import calculator

################################
## Define constants
################################
ymin = 1e-5
day_to_second = 60 * 60 * 24
hour_to_second = 60 * 60 
minute_to_second = 60  

queuetypes = ['fifo', 'priority']
colors = {qtype: color for qtype, color in zip (queuetypes, plt.get_cmap ('Set2').colors)}
colors['diseased'] = '#1f78b4' 
colors['non-diseased'] = '#b2df8a'
colors['interrupting'] = '#b41f78' 
colors['positive'] = '#1f78b4' 
colors['negative'] = '#78b41f'
for gtype, color in zip (['TP', 'FN', 'TN', 'FP'], plt.get_cmap ('Set2').colors):
    colors[gtype] = color
colors['theory'] = 'magenta'
colors['simulation'] = 'darkgray'

################################
## Define lambdas
################################
convert_time = lambda time, time0: (time - time0).total_seconds() / hour_to_second

get_n_positive_patients = lambda oneSim, qtype:len (oneSim.get_positive_records(qtype))
get_n_negative_patients = lambda oneSim, qtype:len (oneSim.get_negative_records(qtype))
get_n_interrupting_patients = lambda oneSim, qtype:len (oneSim.get_interrupting_records(qtype))

################################
## Define plotting functions
################################
## +------------------------------------------
## | For timing of cases
## +------------------------------------------
def get_pclass (record, qtype, hierDict, AIinfo):

    ## Interrupting patients
    if record.is_interrupting: return 'I'

    ## If fifo, no AI positive/negative classes
    if qtype == 'fifo': return 'nonI'

    ## If priority qtype, all AI-positive are considered as positive
    if qtype == 'priority':
        if record.is_positive and record.is_diseased: return 'TP'
        if record.is_positive and not record.is_diseased: return 'FP'
        if not record.is_positive and record.is_diseased: return 'FN'
        if not record.is_positive and not record.is_diseased: return 'TN'

    ## If hierarchical, the condition and AI matters
    for rank, (ainame, _) in enumerate (dict(sorted(hierDict.items(), key=lambda item: item[1])).items()):
        if ainame is None: continue
        diseasename = AIinfo[ainame]['targetDisease']

        if record[ainame] and record.disease_name==diseasename: return 'TP-{0}'.format (rank+1)
        if record[ainame] and record.disease_name!=diseasename: return 'FP-{0}'.format (rank+1)

    if record.is_diseased: return 'FN'
    return 'TN'

def plot_timing (outFile, records, time0, hierDict, AIinfo, n=200, qtype='fifo'):
    
    ''' Function to plot image timestamp workflows for the first N cases.

        inputs
        ------
        outFile (str): filename of the output plot including path
        records (df): dataframe of all simulated patients
        time0 (pandas Timestamp): simulation start time
        n (int): number of cases from the beginning of simulation to be included 
        qtype (str): either 'fifo' or 'priority' to be plotted
    '''

    ## Extract the first N number of cases (default 200)
    records = records[:n]
        
    ## Set up canvas
    h  = plt.figure (figsize=(15, 15))
    gs = gridspec.GridSpec (1, 1)
    gs.update (bottom=0.1)
    axis = h.add_subplot (gs[0])

    ## Extract timestamps (patient arrival trigger time, radiologist open and class time)
    ## and the doctor that is reading the case
    xvalues, yvalues, trigger, close = [], [], [], []
    caseIDs = {'text':[], 'yvalue':[], 'xvalue':[]}
    for index, record in records.iterrows():
        pclass = get_pclass (record, qtype, hierDict, AIinfo)
        caseIDs['text'].append ('#{0} ({1})'.format (index, pclass))
        caseIDs['yvalue'].append (len (records) - index)
        caseIDs['xvalue'].append (convert_time (record[qtype + '_trigger'], time0))
        for n in range (len (record[qtype + '_open'])):
            openTime, closeTime = record[qtype + '_open'][n], record[qtype + '_close'][n]
            xvalues.append (convert_time (openTime, time0))
            yvalues.append (index)
            begin = record[qtype + '_trigger'] if n==0 else record[qtype + '_close'][n-1]
            trigger.append (convert_time (begin, time0))
            close.append (convert_time (closeTime, time0))     
    # Time bar from patient image arrival to radiologist start reading
    xlower = [x-t for t, x in zip (trigger, xvalues)]
    axis.errorbar (xvalues, len (records) - numpy.array (yvalues), marker=None, color='blue',
                   xerr=[xlower, numpy.zeros(len (xvalues))], label='waiting',
                   elinewidth=2, alpha=0.5, ecolor='blue', linestyle='')
    for x, y, t in zip (caseIDs['xvalue'], caseIDs['yvalue'], caseIDs['text']):
        axis.text (x, y, t, horizontalalignment='right', verticalalignment='center', color='black', fontsize=2)
    # Time bar from radiologist start reading to radiologist closing the case
    xupper = [c-x for c, x in zip (close, xvalues)]
    axis.errorbar (xvalues, len (records) - numpy.array (yvalues), marker=None, color='red',
                   xerr=[numpy.zeros(len (xvalues)), xupper], label='serving',
                   elinewidth=2, alpha=0.5, ecolor='red', linestyle='')

    ## Format plotting style
    #  x-axis
    axis.set_xlim (0, numpy.ceil (close[-1]))
    axis.set_xlabel ('Time lapse from start time [hr]', fontsize=15)
    axis.tick_params (axis='x', labelsize=12)
    for xtick in axis.get_xticks():
        axis.axvline (x=xtick, color='gray', alpha=0.3, linestyle='--', linewidth=0.3)
    #  y-axis
    axis.set_ylim (0, len (records)+2)
    for ytick in axis.get_yticks():
        axis.axhline (y=ytick, color='gray', alpha=0.3, linestyle='--', linewidth=0.3)
    axis.get_yaxis().set_ticks([])
    #  legend and title
    axis.legend (loc=1, ncol=1, prop={'size':15})
    axis.set_title ('Timing of top {0} cases in {1}'.format (len (records), qtype), fontsize=15)

    h.savefig (outFile)
    plt.close('all')
    
## +------------------------------------------
## | For state prob distributions
## +------------------------------------------
def get_stats (values, weights=None):

    ''' Obtain statistics from a distribution of input values.
        Statistics include 95% C.I. and 1 sigma ranges, as
        well as median and mean.
        
        inputs
        ------
        values (array): values from which a distribution is built
        weights (array): weights of elements in `values`. Must
                            have same length as `values`.
                            
        output
        ------
        stats (dict): statistics of the distribution from values.
                        Keys: 'lower_95cl', 'lower_1sigma', 'median',
                            'mean', 'upper_1sigma', 'upper_95cl'
    '''

    ## If no valid values, return 0's
    sample = numpy.array(values)[numpy.isfinite(values)]
    if len (sample) == 0:
        return {'lower_95cl':0, 'lower_1sigma':0, 'median':0,
                'upper_1sigma':0, 'upper_95cl':0, 'mean':0}
    
    ## Massage input values and apply weights 
    indices = numpy.argsort (sample)
    sample = numpy.array ([sample[i] for i in indices])
    if weights is None: weights = numpy.ones (len (sample))
    w = numpy.array ([weights[i] for i in indices])
        
    ## Extract statistics from cumulative PDF
    stat = {}
    cumulative = numpy.cumsum(w)/numpy.sum(w)
    stat['lower_95cl']   = sample[cumulative>0.025][0] # lower part is at 50%-47.5% = 2.5%        
    stat['lower_1sigma'] = sample[cumulative>0.16][0]  # lower part is at 50%-34% = 16%
    stat['median']       = sample[cumulative>0.50][0]  # median
    stat['mean']         = numpy.average (sample, weights=w)
    stat['upper_1sigma'] = sample[cumulative>0.84][0]  # upper is 50%+34% = 84%
    stat['upper_95cl']   = sample[cumulative>0.975][0] # upper is 50%+47.5% = 97.5%
    
    return stat

def plot_sim_theory_diseased (outFile, diseases, results):

    ## Set up canvas
    h  = plt.figure (figsize=(7.5, 7.5))
    gs = gridspec.GridSpec (2, 1, height_ratios=[4, 1])
    #gs.update (bottom=0.1)

    ## Top plot: priority and hierarchical wait-time per condition
    axis = h.add_subplot (gs[0])

    yrange = [numpy.inf, -numpy.inf]
    for qtype in ['fifo', 'priority', 'hierarchical']:
        filtered_df = results[results.index.str.contains(qtype)]
        ## For fifo (without-CADt), only one data point
        if qtype == 'fifo':
            ylower = filtered_df.loc['fifo_non-interrupting', 'sim_waittime_low']
            yupper = filtered_df.loc['fifo_non-interrupting', 'sim_waittime_high']
            yvalue = filtered_df.loc['fifo_non-interrupting', 'sim_waittime']
            yerror = [[yvalue - ylower], [yupper - yvalue]]
            axis.errorbar (1, filtered_df.loc['fifo_non-interrupting', 'sim_waittime'],
                           marker='x', ms=10, color='blue',
                           yerr=yerror, label='w/o CADt',
                           elinewidth=2, alpha=0.7, ecolor='blue', linestyle='')      
            axis.scatter (1, filtered_df.loc['fifo_non-interrupting', 'theory_waittime'],
                          marker='*', s=100, facecolors='none', edgecolors='blue', alpha=0.7) 
            if ylower < yrange[0]: yrange[0] = ylower
            if yupper > yrange[1]: yrange[1] = yupper
            continue
        ## For priority/hierarchical, it is per-condition
        xvalues, svalues, slowers, suppers, tvalues = [], [], [], [], []
        color = 'red' if qtype=='priority' else 'green'
        offset = -0.15 if qtype == 'priority' else +0.15
        for dindex, disease in enumerate (diseases):
            xvalues.append (dindex+2+offset)
            indexname = qtype + ' ' + disease
            svalues.append (filtered_df.loc[indexname, 'sim_waittime'])
            slowers.append (filtered_df.loc[indexname, 'sim_waittime_low'])
            suppers.append (filtered_df.loc[indexname, 'sim_waittime_high'])
            tvalues.append (filtered_df.loc[indexname, 'theory_waittime'])
        svalues = numpy.array (svalues)
        slowers = numpy.array (slowers)
        suppers = numpy.array (suppers)
        yerror = [svalues - slowers, suppers - svalues]
        axis.errorbar (xvalues, svalues, marker='x', ms=10, color=color,
                       yerr=yerror, label=qtype + '; sim',
                       elinewidth=2, alpha=0.5, ecolor=color, linestyle='')      
        axis.scatter (xvalues, tvalues, marker='*', s=100, facecolors='none',
                      edgecolors=color, alpha=0.7, label=qtype + '; theory') 
        if min (slowers) < yrange[0]: yrange[0] = min (slowers)
        if max (suppers) > yrange[1]: yrange[1] = max (suppers)        

    ## Format plotting style
    #  x-axis
    axis.set_xlim (0, len (diseases)+2)
    xticks = numpy.arange (len (diseases)+2)
    axis.set_xticks (xticks)
    axis.set_xticklabels([])
    for xtick in axis.get_xticks():
        axis.axvline (x=xtick, color='gray', alpha=0.3, linestyle='--', linewidth=0.3)
    #  y-axis
    yrange[1] += 50
    axis.set_ylim (yrange)
    axis.set_ylabel ('Mean wait-time [min]')
    for ytick in axis.get_yticks():
        axis.axhline (y=ytick, color='gray', alpha=0.3, linestyle='--', linewidth=0.3)
    #  legend and title
    axis.legend (loc=1, ncol=2, prop={'size':12})
    axis.set_title ('Simulation & theory', fontsize=15)

    ## Bottom plot: time-savings
    axis = h.add_subplot (gs[1])

    for qtype in ['priority', 'hierarchical']:
        filtered_df = results[results.index.str.contains(qtype)]
        noAI_svalue = results.loc['fifo_non-interrupting', 'sim_waittime']
        noAI_tvalue = results.loc['fifo_non-interrupting', 'theory_waittime']
        ## For priority/hierarchical, it is per-condition
        xvalues, svalues, tvalues = [], [], []
        color = 'red' if qtype=='priority' else 'green'
        offset = -0.15 if qtype == 'priority' else +0.15
        for dindex, disease in enumerate (diseases):
            xvalues.append (dindex+2+offset)
            indexname = qtype + ' ' + disease
            svalues.append (filtered_df.loc[indexname, 'sim_waittime'] - noAI_svalue)
            tvalues.append (filtered_df.loc[indexname, 'theory_waittime'] - noAI_tvalue)
        axis.scatter (xvalues, svalues, marker='x', c=color, s=100, alpha=0.7, label='sim')    
        axis.scatter (xvalues, tvalues, marker='*', s=100, facecolors='none',
                      edgecolors=color, alpha=0.7) 

    ## Format plotting style
    #  x-axis
    axis.set_xlim (0, len (diseases)+2)
    xticks = numpy.arange (len (diseases)+2)
    axis.set_xticks (xticks)
    xticklabels = ['', 'no-AI'] + list (diseases)
    axis.set_xticklabels (xticklabels, fontsize=12)
    axis.tick_params (axis='x', labelsize=12)
    for xtick in axis.get_xticks():
        axis.axvline (x=xtick, color='gray', alpha=0.3, linestyle='--', linewidth=0.3)
    #  y-axis
    #axis.set_ylim (yrange)
    axis.set_ylabel ('time-savings\n[min]')
    for ytick in axis.get_yticks():
        axis.axhline (y=ytick, color='gray', alpha=0.3, linestyle='--', linewidth=0.3)
    #  legend and title
    #axis.legend (loc=1, ncol=1, prop={'size':12})

    h.savefig (outFile)
    plt.close('all')

def plot_n_patient_distribution (axis, npatients, aclass, params, qtype, doLogY=True):
    
    ''' Function to plot the top subplot i.e. n patient distribution. 

        inputs
        ------
        axis (plt.axis): where data will be plotted
        npatients (): 
        aclass (str): either 'interrupting', 'non-interrupting', 'positive', 'negative'
        params (dict): all parameters related to user settings
        qtype (str): either 'fifo' and 'priority'
        doLogY (bool): if true, y-axis will be in log scale.

        output
        ------
        xticks (array): same xtick values as the distribution plot
    '''

    ## Set up subplots and xticks
    hist_bins = numpy.linspace (0, npatients.max(), int (npatients.max()+1))
    xticks = hist_bins[::2] if max (npatients) < 30 else \
             hist_bins[::4] if max (npatients) < 70 else \
             hist_bins[::8] if max (npatients) < 200 else \
             hist_bins[::16] if max (npatients) < 300 else hist_bins[::32]

    #  Simulation
    hist, edges = numpy.histogram (npatients, bins=hist_bins)
    hist_sum = numpy.sum (hist)
    hist = numpy.r_[hist[0], hist]
    yvalues = hist/hist_sum
    axis.plot (edges[:-1], yvalues[1:], label='sim', color=colors['simulation'], drawstyle='steps-mid',
                linestyle='-', linewidth=2.0, alpha=0.6)
    yvalues = hist[1:]/hist_sum
    yerrors = numpy.sqrt (hist[1:])/hist_sum
    axis.errorbar (edges[:-1], yvalues, marker=None, color=colors['simulation'], yerr=yerrors,
                    elinewidth=1.5, alpha=0.5, ecolor=colors['simulation'], linestyle='')
    #  Theory
    xvalues, yvalues = calculator.get_theory_qlength_fifo_preresume (aclass, qtype, params)
    if yvalues is not None:
        axis.plot (xvalues, yvalues, label='theory', color=colors['theory'], linestyle='--', linewidth=2.0, alpha=0.7)
                            
    # Format x-axis
    axis.set_xlim (xticks[0], xticks[-1])
    axis.set_xticks (xticks)
    axis.set_xticklabels ([int (x) for x in xticks], fontsize=7)
    for xtick in axis.get_xticks():
        axis.axvline (x=xtick, color='gray', alpha=0.3, linestyle='--', linewidth=0.3)
    axis.get_xaxis().set_ticks([])
    # Format y-axis
    if doLogY: axis.set_yscale("log")
    axis.set_ylim (min (hist/hist_sum), max (hist/hist_sum) + 0.05)
    axis.set_ylabel ('Normalized counts', fontsize=9)
    for ytick in axis.get_yticks():
        axis.axhline (y=ytick, color='gray', alpha=0.3, linestyle='--', linewidth=0.3)
    # Format others
    axis.legend (loc='best', ncol=1, prop={'size':9})
    if qtype == 'fifo': qtype = 'Without CADt '
    if qtype == 'priority': qtype = 'With CADt (priority) '
    axis.set_title (qtype + aclass, fontsize=9)

    return xticks

def plot_n_patient_stats (axis, xticks, substats, aclass, params, qtype):
    
    ''' Function to plot the bottom subplot i.e. mean and 95% range of n patient
        distribution. 

        inputs
        ------
        axis (plt.axis): where data will be plotted
        xticks (array): same xtick values as the distribution plot
        substats (dict): stats of n patients for this class and this queue type
        aclass (str): either 'interrupting', 'non-interrupting', 'positive', 'negative'
        params (dict): all parameters related to user settings
        qtype (str): either 'fifo' and 'priority'
    '''

    #  Simulation
    xlower = [substats['mean'] - substats['lower_95cl']]
    xupper = [substats['upper_95cl'] - substats['mean']]
    axis.errorbar (substats['mean'], 1, marker="x", markersize=10, color=colors['simulation'],
                    xerr=[xlower, xupper], elinewidth=3, alpha=0.8, ecolor=colors['simulation'], linestyle='')
    
    ## Theory
    xvalues, yvalues = calculator.get_theory_qlength_fifo_preresume (aclass, qtype, params)
    if yvalues is not None:
        theoryStats = get_stats (xvalues, yvalues)
        xlower = [theoryStats['mean'] - theoryStats['lower_95cl']]
        xupper = [theoryStats['upper_95cl'] - theoryStats['mean']]
        axis.errorbar (theoryStats['mean'], 2, marker="x", color=colors['theory'], markersize=10,
                       xerr=[xlower, xupper], elinewidth=3, alpha=0.8, ecolor=colors['theory'], linestyle='')

    # Format x-axis
    axis.set_xlim (xticks[0], xticks[-1])
    axis.set_xlabel ('number of patients ({0} class only)'.format (aclass), fontsize=9)
    axis.set_xticks (xticks)
    axis.set_xticklabels ([int (x) for x in xticks], fontsize=7)
    for xtick in axis.get_xticks():
        axis.axvline (x=xtick, color='gray', alpha=0.3, linestyle=':', linewidth=0.1)
    # Format y-axis
    axis.set_ylim ([0, 3])
    axis.set_yticks ([1, 2])
    axis.set_yticklabels ([r'sim (95%)', r'theory (95%)'], fontsize=7)
    for ytick in axis.get_yticks():
        axis.axhline (y=ytick, color='gray', alpha=0.3, linestyle=':', linewidth=0.1)

def plot_n_patient_distributions (outFile, nPatients, stats, params):

    ''' Function to plot distributions of the observed number of patients in
        the system *right before a new patient arrives* with CADt. Top row
        is without CADt (interrupting and non-interrupting). Last row is
        with CADt priority queue (interrupting, positive, and negative).

        inputs
        ------
        outFile (str): output plot file name
        nPatients (pandas DataFrame): observed number of patients in system for every patient
                                      typically `trialGen.n_patients_system_df`
        stats (dict): mean, lower/upper 1 sigma and 95% C.I. 
                      typically `trialGen.n_patients_system_stats`
        params (dict): settings for all simulations
    '''

    ## 3 plots (all pre-resume): interrupting, Positive, Negative
    h  = plt.figure (figsize=(25, 12))
    gs = gridspec.GridSpec (2, 3, wspace=0.4, hspace=0.4)

    gindex = 0
    ## Without CADt: Interrupting and Non-interrupting
    for qtype in ['fifo', 'priority']:
        classes = ['interrupting', 'non-interrupting'] if qtype == 'fifo' else \
                  ['interrupting', 'positive', 'negative']
        if qtype == 'priority': gindex = 3
        for aclass in classes:
            ## Skip the plot for interrupting class if fractionED = 0 i.e. no interrupting patients
            if round (params['fractionED'],4) == 0 and aclass == 'interrupting':
                gindex += 1
                continue
            ## Set up subplots and xticks
            subgs = gs[gindex].subgridspec(2, 1, height_ratios=[4, 1], hspace=0.05)        
            ## Top: distribution
            axis = h.add_subplot (subgs[0])
            xticks = plot_n_patient_distribution (axis, nPatients[aclass][qtype], aclass, params, qtype, doLogY=True)
            ## Bottom plot: mean/1sigma
            axis = h.add_subplot (subgs[1])
            plot_n_patient_stats (axis, xticks, stats[qtype][aclass], aclass, params, qtype)
            gindex += 1

    plt.suptitle ('number of total patients in system per class', fontsize=10)
    h.savefig (outFile)
    plt.close('all')

## +------------------------------------------
## | For waiting time distributions
## +------------------------------------------
def plot_waiting_time_distributions (ext, waitTimesDF, stats, params):

    ''' Function to plot distributions of the waiting time and wait-time difference for
        diseased / non-diseased (radiologist diagnosis) and AI positive / negative
        subgroups in both with and without CADt scenarios.

        inputs
        ------
        ext (str): extension of output file name
        waitTimesDF (pandas DataFrame): waiting time for every patient in both with and without
                                        CADt scenario. Typically `trialGen.waiting_times_df`
        stats (dict): mean, lower/upper 1 sigma and 95% C.I. 
                      typically `trialGen.waiting_times_stats`
        params (dict): settings for all simulations
    '''

    ## Calculate time difference per patient between with and without CADt
    waitTimesDF['delta'] = waitTimesDF['priority'] - waitTimesDF.fifo

    ## Plot waiting time grouped by AI call
    outFile = params['plotPath'] + 'waiting_time_distribution_AIcall' + ext
    plot_waiting_time_distribution (outFile, waitTimesDF, stats, params, byDiagnosis=False)

    ## Plot waiting time grouped by radiologist's diagnosis
    outFile = params['plotPath'] + 'waiting_time_distribution_radDiagnosis' + ext
    plot_waiting_time_distribution (outFile, waitTimesDF, stats, params, byDiagnosis=True)

def plot_waiting_time_distribution (outFile, waitTimesDF, stats, params, byDiagnosis=False):

    ''' Function to plot distributions of the waiting time and wait-time difference.

        inputs
        ------
        outFile (str): output plot file name
        waitTimesDF (pandas DataFrame): waiting time for every patient in both with and without
                                        CADt scenario. Typically `trialGen.waiting_times_df`
        stats (dict): mean, lower/upper 1 sigma and 95% C.I. 
                      typically `trialGen.waiting_times_stats`
        params (dict): settings for all simulations
        byDiagnosis (bool): If true, plot diseased and non-diseased (radiologist diagnosis)
                            subgroups. If False, plot AI positive and negative subgroups.
    '''

    #########################################################
    ## If by AI Call: 
    ## +-------------|----------------|---------------+
    ## | Int w/o AI  | non-int w/o AI |    (empty)    |
    ## +-------------|----------------|---------------+
    ## | Int w/ AI   |      AI +      |      AI -     |
    ## +-------------|----------------|---------------+
    ## | Int delta   |   AI + delta   |   AI - delta  |
    ## +-------------|----------------|---------------+
    ##
    ## If by radiologist diagnosis: 
    ## +-------------|----------------|--------------------+
    ## | Int w/o AI  | non-int w/o AI |       (empty)      |
    ## +-------------|----------------|--------------------+
    ## | Int w/ AI   |    Diseased    |     Non-diseased   |
    ## +-------------|----------------|--------------------+
    ## | Int delta   | Diseased delta | Non-diseased delta |
    ## +-------------|----------------|--------------------+   
    #########################################################

    h  = plt.figure (figsize=(20, 20))
    gs = gridspec.GridSpec (3, 3, wspace=0.2, hspace=0.2)
    gs.update (bottom=0.1)

    gindex = 0
    ## Looping through rows
    for qtype in ['fifo', 'priority', 'delta']:

        gtypes = ['interrupting', 'non-interrupting', ''] if qtype == 'fifo' else \
                 ['interrupting', 'diseased', 'non-diseased'] if byDiagnosis else \
                 ['interrupting', 'positive', 'negative'] 
        xlabel = r'$\delta$ waiting time (With CADt - Without CADt) [min]' if gindex>5 else 'waiting time [min]'

        # For a row, loop through column
        for gtype in gtypes:

            # skip empty plot on top right
            ignore = params['fractionED'] == 0 and gtype == 'interrupting'
            if gtype == '' or ignore:
                gindex += 1
                continue

            xticks = [-250, -200, -150, -100, -50, 0, 50] if qtype=='delta' and gtype == 'positive' else \
                     [-200, -100, 0, 100, 200] if qtype=='delta' and gtype in ['diseased', 'non-diseased'] else \
                     [-50, 0, 50, 100, 150, 200, 250] if qtype=='delta' and gtype == 'negative' else \
                     [-50, -25, 0, 25, 50] if qtype=='delta' else \
                     [0, 10, 20, 30, 40, 50] if gtype == 'interrupting' else \
                     [0, 25, 50, 75, 100] if gtype == 'positive' else \
                     [0, 100, 200, 300, 400] ## non-interrupting or negative or diseased or non-diseased
            hist_bins = numpy.linspace (xticks[0], xticks[-1], 21)

            subgs = gs[gindex].subgridspec(2, 1, height_ratios=[4, 1], hspace=0.05) 

            # +----------------------------------------------
            # | Top: distributions
            # +----------------------------------------------
            axis = h.add_subplot (subgs[0])
            # Simulation
            if gtype == 'interrupting':
                values = waitTimesDF[qtype][waitTimesDF.is_interrupting]
            else:
                nonInterrupting = waitTimesDF[~waitTimesDF.is_interrupting]
                values = nonInterrupting[qtype]
                if gtype == 'positive': values = values[nonInterrupting.is_positive]
                if gtype == 'negative': values = values[~nonInterrupting.is_positive]
                if gtype == 'diseased': values = values[nonInterrupting.is_diseased]
                if gtype == 'non-diseased': values = values[nonInterrupting.is_diseased==False]

            hist, edges = numpy.histogram (values, bins=hist_bins)
            hist_sum = numpy.sum (hist)
            hist = numpy.r_[hist[0], hist]
            yvalues = hist/hist_sum
            axis.plot (edges[:-1], yvalues[1:], label='sim', color=colors['simulation'], drawstyle='steps-mid',
                       linestyle='-', linewidth=2.0, alpha=0.6)
            yvalues = hist[1:]/hist_sum
            yerrors = numpy.sqrt (hist[1:])/hist_sum
            axis.errorbar (edges[:-1], yvalues, marker=None, color=colors['simulation'], yerr=yerrors,
                        elinewidth=1.5, alpha=0.5, ecolor=colors['simulation'], linestyle='')

            # Theory
            # No theoretical predictions for the waiting time distributions. Only means are predicted.

            # If delta time, add a vertical line at x = 0
            if qtype=='delta': axis.axvline (x=0, color='black', linestyle='-', linewidth=1.0, alpha=0.6)

            # Format x-axis
            axis.set_xlim (xticks[0], xticks[-1])
            axis.set_xticks (xticks)
            axis.set_xticklabels ([int (x) for x in xticks], fontsize=10)
            for xtick in axis.get_xticks():
                axis.axvline (x=xtick, color='gray', alpha=0.3, linestyle='--', linewidth=0.3)
            axis.get_xaxis().set_ticks([])
            # Format y-axis
            axis.set_yscale("log")
            axis.set_ylim (ymin, max (hist/hist_sum) + 10)    
            axis.set_ylabel ('Normalized counts', fontsize=12)
            for ytick in axis.get_yticks():
                axis.axhline (y=ytick, color='gray', alpha=0.3, linestyle='--', linewidth=0.3)
            # Format others
            axis.set_title ('{0} patients ({1})'.format (gtype, qtype), fontsize=12)

            # +----------------------------------------------
            # | Bottom: Mean / CI
            # +----------------------------------------------
            axis = h.add_subplot (subgs[1])

            #  Simulation
            thisStats = stats['fifo']['waitTime'][gtype] if qtype == 'fifo' else \
                        stats['priority']['waitTime'][gtype] if qtype == 'priority' else \
                        stats['priority']['diff'][gtype]
            xlower = [max (0, thisStats['mean'] - thisStats['lower_95cl'])]
            xupper = [max (0, thisStats['upper_95cl'] - thisStats['mean'])]
            axis.errorbar (thisStats['mean'], 1, marker="x", markersize=10, color=colors['simulation'], label='sim',
                           xerr=[xlower, xupper], elinewidth=3, alpha=0.8, ecolor=colors['simulation'], linestyle='')

            # Plot theoretical mean
            theory = calculator.get_theory_waitTime (gtype, qtype, params)
            if theory is not None:
                axis.scatter (theory, 2, s=60, marker="+", color=colors['theory'], alpha=0.5, label='theory')

            # If delta time, add a vertical line at x = 0
            if qtype=='delta': axis.axvline (x=0, color='black', linestyle='-', linewidth=1.0, alpha=0.6)

            # Format x-axis
            axis.set_xlim (xticks[0], xticks[-1])
            axis.set_xlabel (xlabel, fontsize=12)
            axis.set_xticks (xticks)
            axis.set_xticklabels ([int (x) for x in xticks], fontsize=10)
            for xtick in axis.get_xticks():
                axis.axvline (x=xtick, color='gray', alpha=0.3, linestyle='--', linewidth=0.3)
            # Format y-axis
            axis.set_ylim ([0, 3])
            axis.set_yticks ([1, 2])
            axis.set_yticklabels ([r'sim (95%)', r'theory'], fontsize=7)
            for ytick in axis.get_yticks():
                axis.axhline (y=ytick, color='gray', alpha=0.3, linestyle='--', linewidth=0.3)
            # Format others
            axis.legend (loc=1, ncol=1, prop={'size':6})

            gindex += 1

    group = 'radiologist diagnosis' if byDiagnosis else 'AI call'
    plt.suptitle ('Distributions of waiting time and wait-time difference by {0}'.format (group), fontsize=15)
    h.savefig (outFile)
    plt.close('all')
