import os
import pkg_resources
import json
from collections import defaultdict


UNIGRAM_FILE = pkg_resources.resource_filename("convokit",
    os.path.join("politeness_local/lexicons", "unigram_strategies.json"))

NGRAM_FILE = pkg_resources.resource_filename("convokit",
    os.path.join("politeness_local/lexicons", "ngram_strategies.json"))


START_FILE = pkg_resources.resource_filename("convokit",
    os.path.join("politeness_local/lexicons", "start_strategies.json"))

MODE_FILE = pkg_resources.resource_filename("convokit",
    os.path.join("politeness_local/lexicons", "marker_mode.json"))


############## LOADING MARKER INFO ####################

# Loading basic markers 
def load_basic_markers(unigram_path=UNIGRAM_FILE, ngram_path=NGRAM_FILE, start_path=START_FILE):
    
    # load unigram markers 
    with open(unigram_path, "r") as f:
        unigram_dict = json.load(f)

    with open(ngram_path, "r") as f:
        ngram_dict = json.load(f)

    # load phrase markers 
    with open(start_path, "r") as f:
        start_dict = json.load(f)
    
    return unigram_dict, start_dict, ngram_dict


def load_marker_mode(mode_path=None):
    
    if mode_path is None:
        mode_path = MODE_FILE
    
    with open(mode_path, "r") as f:
        marker_mode = json.load(f)
        
    return marker_mode