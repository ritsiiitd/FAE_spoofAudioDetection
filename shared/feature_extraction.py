import numpy as np
import soundfile as sf
from spafe.features.spfeats import extract_feats
from pydub.utils import mediainfo
from shared.constants import DROP_FEATURES, SP_FEATS_NAMES, SPECTRAL_COMPLEX_VALUES, SPECTRUM_FEATURES

from spafe.features.mfcc import mfcc, imfcc
from spafe.features.bfcc import bfcc
from spafe.features.lfcc import lfcc
from spafe.features.lpc import lpc, lpcc
from spafe.features.msrcc import msrcc
from spafe.features.ngcc import ngcc
from spafe.features.psrcc import psrcc
from spafe.features.rplp import plp, rplp
from spafe.features.gfcc import gfcc

SPECTRUM_FEATURES_FUNCTIONS = [mfcc, imfcc, bfcc, lfcc, lpc, lpcc, msrcc, ngcc, psrcc, plp, rplp, gfcc]

def extract_sp_feats(file_path: str, dtype: str = "float64") -> dict:
    sig, fs = sf.read(file_path, dtype=dtype)
    sp_feats = extract_feats(sig=sig, fs=fs)

    # apply mean to arrays
    for sp_feat_name in SP_FEATS_NAMES:
        sp_feat_type = type(sp_feats[sp_feat_name])
        if sp_feat_type == tuple or sp_feat_type == np.ndarray or sp_feat_type == list:
            sp_feats[sp_feat_name] = np.array(sp_feats[sp_feat_name])
            sp_feats[sp_feat_name] = sp_feats[sp_feat_name].mean() if len(sp_feats[sp_feat_name]) > 0 else 0
        elif sp_feat_name in SPECTRAL_COMPLEX_VALUES:
            sp_feats[sp_feat_name] = np.array(sp_feats[sp_feat_name]).real.mean()

    return sp_feats

def extract_media_info(file_path: str, selected_features=['bit_rate']) -> dict:
    media_info = mediainfo(file_path)
    features = {}
    for f in selected_features:
        features[f] = media_info[f] if f in media_info else 0
    return features


def extract_spectrum_data(sample: str) -> dict:
    sig, fs = sf.read(sample, dtype="float64")

    spectrum_dict = {}
    spectrum_dict["signal"] = sig.mean()
    for idx in range(len(SPECTRUM_FEATURES)):
        feature = SPECTRUM_FEATURES_FUNCTIONS[idx](sig=sig, fs=fs)
        spectrum_dict[SPECTRUM_FEATURES[idx]] = feature.mean()

    return spectrum_dict

def filter_features(features: dict):
    filtered_features = {}
    for f in features:
        if f not in DROP_FEATURES:
            filtered_features[f] = features[f]

    return list(filtered_features.values())


def get_all_features_from_sample(file_path: str):
    sp_feats = extract_sp_feats(file_path)
    media_info = extract_media_info(file_path)
    spectrum_data = extract_spectrum_data(file_path)
    features = {}
    features.update(sp_feats)
    features.update(media_info)
    features.update(spectrum_data)
    ret = filter_features(features)
    ret2=[]
    
    for i in range(len(ret)):
        if isinstance(ret[i], str):
            ret[i] = np.float64(ret[i])
        
    for i in range(len(ret)):
        # print(i,ret[i],type(ret[i]),np.isnan(ret[i]))
        if(np.isnan(ret[i])):
            ret2.append(0.0)
        else:
            ret2.append(ret[i])

        
    # print(ret)
    print(ret2)
    return ret2

