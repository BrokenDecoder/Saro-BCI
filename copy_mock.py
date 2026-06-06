import shutil, os
src = r"d:\Saro Bci And BioMechanics\data\chb01_01.edf"
base = r"d:\Saro Bci And BioMechanics\Saro Bci Project\data\raw"
files = [
    r"zeropad_30\individual\nt9_speak_zeropad_30_components_preprocessed.edf",
    r"zeropad_30\individual\nt10_listen_zeropad_30_components_preprocessed.edf",
    r"zeropad_30\individual\nt9_listen_zeropad_30_components_preprocessed.edf",
    r"zeropad_30\individual\nt10_speak_zeropad_30_components_preprocessed.edf",
    r"zeropad_30\stacked\nt9_zeropad_speak_listen_stacked.edf",
    r"zeropad_30\stacked\nt10_zeropad_listen_speak_stacked.edf",
    r"cut_60\individual\nt9_speak_cut_60_components_preprocessed.edf",
    r"cut_60\individual\nt10_listen_cut_60_components_preprocessed.edf",
    r"cut_60\individual\nt9_listen_cut_60_components_preprocessed.edf",
    r"cut_60\individual\nt10_speak_cut_60_components_preprocessed.edf",
    r"cut_60\stacked\nt9_cut_speak_listen_stacked.edf",
    r"cut_60\stacked\nt10_cut_listen_speak_stacked.edf"
]
for f in files:
    full = os.path.join(base, f)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    shutil.copy(src, full)
