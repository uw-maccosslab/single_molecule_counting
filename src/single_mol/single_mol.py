import pymzml
import pandas as pd
import msions.utils as msutils
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import msions.kronik as kro
import sqlite3
import msions.encyclopedia as encyclo
import msions.mzml as mzml
import msions.hardklor as hk
from matplotlib.ticker import FormatStrFormatter  # format decimal places
from typing import List


def fig2a(mzml_file: str, bin_rt_list: List[float], bin_mz_list: List[float], manu_mzml: bool = False):
    """
    Creates the Figure 2a heatmap using an mzML file and binning lists.

    Parameters
    ----------
    mzml_file : str
        The input mzML file.
    bin_rt_list : List[float]
        List of retention time bin edges.
    bin_mz_list : List[float]
        List of m/z bin edges.
    manu_mzml: bool
        Boolean whether the mzML was used in the manuscript
        to determine if a red line should be plotted like 
        in the manuscript.	

    Examples
    -------
    >>> import msions.utils as ustils
    >>> from single_mol import fig2a
    >>> rt_bin_size = 0.25
    >>> rt_bin_mult = 1
    >>> rt_start = 0
    >>> rt_end = 100
    >>> bin_rt_list = msutils.bin_list(rt_start, rt_end, rt_bin_size, rt_bin_mult)
    >>> mz_bin_size = 4
    >>> mz_bin_mult = 1.0005
    >>> mz_start = 399
    >>> mz_end = 1005
    >>> bin_mz_list = msutils.bin_list(mz_start, mz_end, mz_bin_size, mz_bin_mult)
    >>> fig2a("test.mzML", bin_rt_list, bin_mz_list, manu_mzml=True)
    """    
    # create DataFrame of MS1 peaks
    peak_df = mzml.peak_df(mzml_file)

    # create DataFrame with binned m/z and retention time
    peak_binned = msutils.bin_data(peak_df.copy(), type="both", bin_mz_list=bin_mz_list, bin_rt_list=bin_rt_list)

    # create a max cut-off to help with dynamic range
    max_binned = peak_binned.copy()
    max_binned["ips_max"] = [x if x < 2e9 else 2e9 for x in max_binned["ips"]]

    # create a pivot table to make the heatmap
    max_pivot_table = pd.pivot_table(max_binned, values='ips_max', index=['bin_mz'],
                                    columns=['bin_rt'], aggfunc=np.sum)

    # sort the m/z to make 400 closer to the x-axis
    max_pivot_table = max_pivot_table.sort_values(by=['bin_mz'], ascending=False)    

    # heatmap for data
    fig, ax = plt.subplots(figsize=(10,8))

    # create heatmap, remove color bar
    # remove last 3 columns because there was no data for those bins
    sns.heatmap(max_pivot_table.iloc[:, :-3].astype('float'), cmap='Greys', cbar=False)

    if manu_mzml:
        # create a line to correspond to (b)
        plt.vlines(x=230+0.17/0.25, ymin=0, ymax=1000, color="red", linestyles='dashed', linewidth=2)

    # set x-axis parameters
    plt.xlabel('Retention Time (min)',fontsize=18)
    ax.set_xticks((range(0, len(max_pivot_table.columns), math.floor(len(max_pivot_table.columns)/5))))
    ax.set_xticklabels((0,20,40,60,80,100), rotation=0, fontsize=14)
    ax.axhline(y=151, color='black', linewidth=2)

    # set y-axis parameters
    plt.ylabel('m/z',fontsize=18)
    ax.set_yticks((range(0, len(max_pivot_table.index), math.floor(len(max_pivot_table.index)/6))))
    ax.set_yticklabels((1000,900,800,700,600,500,400), fontsize=14)
    ax.axvline(x=0, color='black', linewidth=1)

    # show plot
    plt.show()


def fig2a_insert(kro_file: str, elib_file: str):
    """
    Creates the Figure 2a histogram insert using a Kronik file and EncyclopeDIA elib.

    Parameters
    ----------
    kro_file : str
        The input Kronik file.
    elib_file : str
        The input EncyclopeDIA elib file.

    Examples
    -------
    >>> from single_mol import fig2a_insert
    >>> fig2a_insert("test.kro", "test.elib")
    """  
    # read in the Kronik output
    kro_df = kro.simple_df(kro_file)

    # create connection
    elib_connection = sqlite3.connect(elib_file)

    # initiate empty lists
    id_list = []
    peptide_list = []

    # create sql query
    sql = """SELECT PeptideSeq
            FROM entries
            WHERE ((? BETWEEN RTInSecondsStart AND RTInSecondsStop) AND ((PrecursorMz - ?) BETWEEN -? AND ?));"""

    # find the peptide(s) associated with each m/z
    for idx, mz in enumerate(kro_df["mz"]):
        rt = kro_df.loc[idx, "best_rt_s"]
        sql_df = pd.read_sql_query(sql, elib_connection, params=(rt, mz, mz*5e-6, mz*5e-6))
        id_list.append(len(sql_df))
        peptide_list.append(sql_df["PeptideSeq"])

    # add identification status and peptide list to DataFrame
    kro_df["id"] = id_list
    kro_df["peptides"] = peptide_list

    # close the database connection
    elib_connection.close()

    # create a bin list for intensities
    # removed last 2 bins that were empty
    bin_int_list = msutils.bin_list(4, 10, 0.2, 1)[:-2]

    # create an array of intensities for all persistent features
    all_int = np.log10(kro_df.best_int)

    # calculate counts of all persistent features in each bin
    all_counts = np.histogram(all_int, bins=bin_int_list)[0]

    # create an array of intensities for identified persistent features
    id_int = np.log10(kro_df[kro_df["id"] > 0].best_int)

    # calculate counts of identified persistent features in each bin
    id_counts = np.histogram(id_int, bins=bin_int_list)[0]

    # define figure size
    fig, ax = plt.subplots(figsize=(10,8))

    # plot histograms
    # all persistent features
    ax.hist(all_int, bins=bin_int_list, histtype="bar", color="dimgray")
    # identified persistent features
    ax.hist(id_int, bins=bin_int_list, histtype="bar", color="#1f77b4", alpha=0.8)

    # set x-axis parameters
    plt.xlabel("log10(Ions/Second)", fontsize=18)
    plt.xticks(fontsize=14)

    # set y-axis parameters
    ax.set_ylabel("Counts",fontsize=18)
    plt.yticks(fontsize=14)

    # create a second axis
    ax2 = ax.twinx()

    # plot line of percent identifed
    # removed last edge to match array lengths
    ax2.plot(bin_int_list[:-1], (id_counts/all_counts)*100, color="red")

    # set second y-axis parameters
    ax2.set_ylabel("Percent Identified", fontsize=18, rotation=270, color="red", labelpad=12)
    ax2.set_ylim(0,101)
    plt.yticks(fontsize=14)

    # despine top
    ax.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    # show plot
    plt.show()


def fig2b(mzml_file, hk_file, elib_file, scan_num=170933, id_peaks=7):
    """
    Creates the Figure 2b plot of identified and unidentified MS1 peaks.

    Also prints the highest intensity identified peaks.

    Parameters
    ----------
    mzml_file : str
        The input mzML file.
    hk_file : str
        The input Hardklor file.
    elib_file : str
        The input EncyclopeDIA elib file.
    scan_num : int
        Scan number of plotted peaks.
    id_peaks : int
        Number of highest intensity identified peaks to print information about.

    Examples
    -------
    >>> from single_mol import fig2b
    >>> fig2b("test.mzML", "test.hk", "test.elib")
    """    
    # create DataFrame from elib
    ev_encyclo_df = encyclo.dia_df(elib_file)

    # create DataFrame from Hardklor output
    ev_hk_df = hk.hk2df(hk_file)

    # find Hardklor retention time/charge/mass match in encyclopeDIA results
    ev_hk_df["in_encyclo"] = ev_hk_df.apply(encyclo.match_hk, axis=1, other_df=ev_encyclo_df)

    # create run object
    run = pymzml.run.Reader(mzml_file)

    # choose MS1 spectrum between 40-60 minutes
    ms1_spectrum = run[scan_num]  # 57.6733 min = 3460.398 sec

    # define retention time
    spec_rt = round(ms1_spectrum.scan_time[0], 4)

    # create pandas DataFrame of MS1 peaks
    ms1_peaks = pd.DataFrame(ms1_spectrum.peaks("centroided")).rename(columns={0: "mz", 1: "ips"})
    ms1_peaks["mz"] = ms1_peaks["mz"].round(4)

    # create narrow hk_df
    peak_hk_df = ev_hk_df[ev_hk_df["rt"] == spec_rt]

    # merge identification designation onto MS1 peaks
    ms1_peaks = pd.merge_asof(ms1_peaks, peak_hk_df[['mz', 'mass', 'charge', 'in_encyclo']], tolerance=0.001, on=['mz'])
    ms1_peaks = ms1_peaks.fillna(-1)

    # write algorithm to color peptide isotope distributions
    # initiate indices
    idx, idx2, idx3 = [0, 0, 0]

    # initiate empty lists for previous values
    prev_idxs = []

    # initiate empty list to be added to DataFrame
    in_dist_list = [0]*len(ms1_peaks)

    # list to go forward through indices:
    fwd_idxs = [x for x in range(0, len(ms1_peaks))]

    # loop through whether MS1 peaks were found in Hardklor and encyclopeDIA
    for idx, val in enumerate(ms1_peaks["in_encyclo"]):
        # if peak was seen in Hardklor and encyclopeDIA,
        if val >= 1:
            # the peak is in a distribution
            in_dist_list[idx] = 1

            # define the mass, mz, and charge of interest
            mono_mass = ms1_peaks.loc[idx, "mass"]
            check_mz = ms1_peaks.loc[idx, "mz"]
            charge_state = ms1_peaks.loc[idx, "charge"]

            # calculate the isotopic difference in mz
            iso_diff = 1/charge_state

            # go back through stored idxs
            for idx2 in reversed(prev_idxs):
                # if the stored idx's mz is within the isotopic difference of the mz of interest
                if np.isclose(ms1_peaks.loc[idx2, "mz"], check_mz-iso_diff, atol=0.01):
                    in_dist_list[idx2] = 1
                    check_mz = ms1_peaks.loc[idx2, "mz"]
                elif (check_mz+iso_diff)-ms1_peaks.loc[idx3, "mz"] > 2*iso_diff:
                    break

            # go forward through upcoming idxs        
            for idx3 in fwd_idxs[idx+1:]:
                # if the stored idx's mz is within the isotopic difference of the mz of interest
                if np.isclose(ms1_peaks.loc[idx3, "mz"], check_mz+iso_diff, atol=0.01):
                    in_dist_list[idx3] = 1
                    check_mz = ms1_peaks.loc[idx3, "mz"]    
                elif ms1_peaks.loc[idx3, "mz"]-(check_mz+iso_diff) > 2*iso_diff:
                    in_dist = False
                    break                

        # if peak was not seen in encyclopeDIA
        else:
            # keep track of seen indices that are not monoisotopic peaks
            prev_idxs.append(idx)

    # add the column to the ms1_peaks DataFrame
    ms1_peaks["in_dist"] = in_dist_list

    # create DataFrames of identified and non-identified peaks
    id_ms1_peaks = ms1_peaks[ms1_peaks["in_dist"] > 0].reset_index(drop=True)
    noid_ms1_peaks = ms1_peaks[ms1_peaks["in_dist"] == 0].reset_index(drop=True)

    # create connection
    elib_connection = sqlite3.connect(elib_file)

    # create empty dictionary and list for results
    mz2pep_dict = {}
    pep_list = []

    # create list of masses
    mz_list = ms1_peaks[ms1_peaks["in_encyclo"] > 0].sort_values(by="ips", ascending=False).head(id_peaks).mz

    # create sql query
    sql = """SELECT PeptideSeq
            FROM entries
            WHERE ((3460.398 BETWEEN RTInSecondsStart AND RTInSecondsStop) AND ((PrecursorMz - ?) BETWEEN -0.01 AND 0.01));"""

    # print the protein(s) associated with each peptide sequence
    for mz in mz_list:
        sql_df = pd.read_sql_query(sql, elib_connection, params=[mz])
        for pep in sql_df["PeptideSeq"]:
            if mz in mz2pep_dict.keys():
                mz2pep_dict[mz].append(pep)
            else:
                mz2pep_dict[mz] = [pep]
            pep_list.append(pep)

    # close the database connection
    elib_connection.close()

    # create connection
    elib_connection = sqlite3.connect(elib_file)

    # initiate empty protein dictionary
    seq2pro_dict = {}

    # create sql query
    sql = """SELECT ProteinAccession 
            FROM peptidetoprotein
            WHERE PeptideSeq = ?;"""

    # print the protein(s) associated with each peptide sequence
    for seq in pep_list:
        sql_df = pd.read_sql_query(sql, elib_connection, params=[seq])
        for protein in sql_df["ProteinAccession"]:
            if seq in seq2pro_dict.keys():
                seq2pro_dict[seq].append(protein)
            else:
                seq2pro_dict[seq] = [protein]		

    # close the database connection
    elib_connection.close()

    # combine dictionary info
    for mz in mz_list:
        for pep in mz2pep_dict[mz]:
            for pro in seq2pro_dict[pep]:
                print(mz, "m/z:", pep, "("+pro+")")

    # define figure size
    plt.figure(figsize=(10,8))

    # plot spectra
    plt.vlines(x= noid_ms1_peaks["mz"], ymin = 0, ymax = noid_ms1_peaks["ips"]/1e8, color="red")
    plt.vlines(x= id_ms1_peaks["mz"], ymin = 0, ymax = id_ms1_peaks["ips"]/1e8, color="#1f77b4")

    # set x-axis parameters
    plt.xlabel("m/z", fontsize=18)
    plt.xlim(395,1000)
    plt.xticks(fontsize=14)

    # set y-axis parameters
    plt.ylabel("Ions/Second ($10^{8}$)",fontsize=18)
    plt.ylim(0,14.5)
    plt.yticks(np.arange(2,15,2))
    plt.yticks(fontsize=14)

    # despine top and right
    sns.despine()

    # show plot
    plt.show()


def fig2c(mzml_file, hk_file, elib_file):
    """
    Creates the Figure 2c plot of Total Ion Current (TIC) 
    and identified ion current.

    Also prints the values and percentage identified.

    Parameters
    ----------
    mzml_file : str
        The input mzML file.
    hk_file : str
        The input Hardklor file.
    elib_file : str
        The input EncyclopeDIA elib file.

    Examples
    -------
    >>> from single_mol import fig2c
    >>> fig2c("test.mzML", "test.hk", "test.elib")
    """    
    # create DataFrame from elib
    ev_encyclo_df = encyclo.dia_df(elib_file)

    # create DataFrame from mzML
    ev_ms1_df = mzml.tic_df(mzml_file)

    # create DataFrame from Hardklor output
    ev_hk_df = hk.hk2df(hk_file)

    #summarize Hardklor DataFrame
    summed_ev_hk = hk.summarize_df(ev_hk_df, full_ms1_df=ev_ms1_df)

    # find Hardklor retention time/charge/mass match in encyclopeDIA results
    ev_hk_df["in_encyclo"] = ev_hk_df.apply(encyclo.match_hk, axis=1, other_df=ev_encyclo_df)

    # create DataFrame of only identified features
    id_ev_hk_df = ev_hk_df[ev_hk_df["in_encyclo"] > 0].reset_index(drop=True)

    # summarize identified features DataFrame
    sum_id_ev_hk_df = hk.summarize_df(id_ev_hk_df, full_ms1_df=ev_ms1_df)

    # join DataFrames
    joined_EV = pd.merge(ev_ms1_df, sum_id_ev_hk_df, how="outer", on=["scan_num"], suffixes=("_ms1","_id"))

    # find total ion current across all 
    print("Total Ion Current (TIC): %.0f" % sum(ev_ms1_df.TIC))

    # find identified total ion current
    print("ID'd TIC: %.0f" % sum(sum_id_ev_hk_df.TIC))

    # calculate ratio of identified ion current to total ion current
    print("%.1f%% of the signal" % float(sum(sum_id_ev_hk_df.TIC)/sum(ev_ms1_df.TIC)*100))

    # create figure and axis objects with subplots()
    fig, ax = plt.subplots(figsize=(10,8))

    # make a TIC plot
    ax.plot(joined_EV.rt_ms1, joined_EV.TIC_ms1/1e10, color="black")
    ax.plot(joined_EV.rt_ms1, joined_EV.TIC_id/1e10, color="#1f77b4")

    # set x-axis parameters
    ax.set_xlabel("Retention Time (min)", fontsize=18)
    ax.set_xlim(0,100)
    plt.xticks(fontsize=14)

    # set y-axis parameters
    ax.set_ylabel("Ions/Second ($10^{10}$)",fontsize=18)
    ax.set_ylim(0,4.5)
    ax.set_yticks(np.arange(1,5,1))
    plt.yticks(fontsize=14)

    # despine top and right
    sns.despine()

    # show plot
    plt.show()


def fig2d(mzml_file, hk_file, elib_file):
    """
    Creates the Figure 2d plot of total ions
    and identified ions.

    Also prints the values and percentage identified.

    Parameters
    ----------
    mzml_file : str
        The input mzML file.
    hk_file : str
        The input Hardklor file.
    elib_file : str
        The input EncyclopeDIA elib file.

    Examples
    -------
    >>> from single_mol import fig2d
    >>> fig2d("test.mzML", "test.hk", "test.elib")
    """    
    # create DataFrame from elib
    ev_encyclo_df = encyclo.dia_df(elib_file)

    # create DataFrame from mzML
    ev_ms1_df = mzml.tic_df(mzml_file)

    # create DataFrame from Hardklor output
    ev_hk_df = hk.hk2df(hk_file)

    #summarize Hardklor DataFrame
    summed_ev_hk = hk.summarize_df(ev_hk_df, full_ms1_df=ev_ms1_df)

    # find Hardklor retention time/charge/mass match in encyclopeDIA results
    ev_hk_df["in_encyclo"] = ev_hk_df.apply(encyclo.match_hk, axis=1, other_df=ev_encyclo_df)

    # create DataFrame of only identified features
    id_ev_hk_df = ev_hk_df[ev_hk_df["in_encyclo"] > 0].reset_index(drop=True)

    # summarize identified features DataFrame
    sum_id_ev_hk_df = hk.summarize_df(id_ev_hk_df, full_ms1_df=ev_ms1_df)

    # join DataFrames
    joined_EV = pd.merge(ev_ms1_df, sum_id_ev_hk_df, how="outer", on=["scan_num"], suffixes=("_ms1","_id"))

    # find total ions across all 
    print("Total # of ions: %.0f" % sum(ev_ms1_df.ions))

    # find identified ions
    print("Ions mapped to peptides: %.0f" % sum(sum_id_ev_hk_df.ions))

    # calculate ratio of identified ions to total ions
    print("%.1f%% of the signal" % float(sum(sum_id_ev_hk_df.ions)/sum(ev_ms1_df.ions)*100))

    # create figure and axis objects with subplots()
    fig, ax = plt.subplots(figsize=(10,8))

    # make a plot for ion counts
    ax.plot(joined_EV.rt_ms1, joined_EV.ions_ms1/1e6, color="black")
    ax.plot(joined_EV.rt_ms1, joined_EV.ions_id/1e6, color="#1f77b4")

    # set x-axis parameters
    ax.set_xlabel("Retention Time (min)", fontsize=18)
    ax.set_xlim(0,100)
    plt.xticks(fontsize=14)

    # set y-axis parameters
    ax.set_ylabel("Ions ($10^{6}$)",fontsize=18)
    ax.set_ylim(0,6.1)
    ax.set_yticks(np.arange(1,7,1))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    plt.yticks(fontsize=14)

    # despine top and right
    sns.despine()

    # show plot
    plt.show()
