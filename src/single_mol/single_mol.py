import pymzml
import pandas as pd
import msions.utils as utils

import msions.encyclopedia as encyclo
import msions.mzml as mzml
import msions.hardklor as hk


def fig2a(mzml_file, bin_mz_list, bin_rt_list):
    # create run object
    run = pymzml.run.Reader("panoramaweb_files/plasma-EV-01.mzML")

    # initiate peak dictionary
    peak_dict = {}

    # initiate peak DataFrame
    peak_df = pd.DataFrame(columns=["mz", "ips", "rt"])

    # loop through spectra
    for spectra in run:
        if spectra.ms_level == 1:
            peak_array = pd.DataFrame(spectra.peaks("centroided")).rename(columns={0: "mz", 1: "ips"})
            peak_array["mz"] = peak_array["mz"].round(4)
            peak_array["rt"] = spectra.scan_time[0]
            peak_df = pd.concat([peak_df, peak_array])
            peak_dict[spectra.ID] = peak_array


def fig2c():
    # create DataFrame from elib
    ev_encyclo_df = encyclo.dia_df("panoramaweb_files/plasma-EV-01.mzML.elib")

    # create DataFrame from mzML
    ev_ms1_df = mzml.tic_df("panoramaweb_files/plasma-EV-01.mzML")

    # create DataFrame from Hardklor output
    ev_hk_df = hk.hk2df("panoramaweb_files/plasma-EV-01_MS1_3sn.hk")

    #summarize Hardklor DataFrame
    summed_ev_hk = hk.summarize_df(ev_hk_df, full_ms1_df=ev_ms1_df)

    # find Hardklor retention time/charge/mass match in encyclopeDIA results
    ev_hk_df["in_encyclo"] = ev_hk_df.apply(encyclo.match_hk, axis=1, other_df=ev_encyclo_df)

    # create DataFrame of only identified features
    id_ev_hk_df = ev_hk_df[ev_hk_df["in_encyclo"] > 0].reset_index(drop=True)

    # summarize identified features DataFrame
    sum_id_ev_hk_df = hk.summarize_df(id_ev_hk_df, full_ms1_df=ev_ms1_df)

    # find total ion current across all 
    print("Total Ion Current (TIC): %.0f" % sum(ev_ms1_df.TIC))

    # find identified total ion current
    print("ID'd TIC: %.0f" % sum(sum_id_ev_hk_df.TIC))

    # calculate ratio of identified ion current to total ion current
    print("%.1f%% of the signal", sum(sum_id_ev_hk_df.TIC)/sum(ev_ms1_df.TIC)*100)