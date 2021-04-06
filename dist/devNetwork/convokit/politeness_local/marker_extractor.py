from itertools import chain 
from collections import defaultdict
from convokit.politeness_local.marker_loader import load_basic_markers, load_marker_mode

# load markers 
UNIGRAM_MARKERS, START_MARKERS, NGRAM_MARKERS = load_basic_markers()
MARKER_MODE = load_marker_mode()

############# Helper functions and variables #################

def extract_unigram_markers(sent_parsed, sent_idx, unigram_markers):
    
    return [(info['tok'], sent_idx, idx) for idx, info in enumerate(sent_parsed) if info['tok'].lower() in unigram_markers]


def extract_ngram_markers(sent_parsed, sent_idx, ngram_markers):
    
    ngrams_used = []
    words = [info['tok'].lower() for info in sent_parsed]
        
    for i, info in enumerate(sent_parsed[0:-1]):
        
        for ngram in ngram_markers:
            
            ngram_words = ngram.split()

            if words[i:i+len(ngram_words)] == ngram_words:
                
                start_idx = i
                ngrams_used.extend([(tok, sent_idx, start_idx+k) for k, tok in enumerate(ngram_words)])
                
    return ngrams_used


def extract_starter_markers(sent_parsed, sent_idx, starter_markers):
    
    start_tok = sent_parsed[0]['tok'].lower()
    
    if start_tok in starter_markers:
        return [(start_tok, sent_idx, 0)]
    else:
        return []
    
    
def extract_dep_parse_markers(sent_parsed, sent_idx, child, parents=None, relation=None):
    
    matched = []
    
    for i, tok in enumerate(sent_parsed):            
        
        # if tok doesn't match
        if tok['tok'].lower() != child:
            continue 
        
        # if relation doesn't match
        if relation is not None and tok['dep'] != relation:
            continue 
        
        # if parant doesn't match
        if parents is not None and (tok['dep'] == 'ROOT' or sent_parsed[tok['up']]['tok'].lower() not in parents):
            continue
        
        # if parent is not specified, only track child 
        if parents is None: 
            matched.append((tok['tok'], sent_idx, i))
            
        # keep both child and parent
        else:
            matched.extend([(tok['tok'], sent_idx, i), (sent_parsed[tok['up']]['tok'], sent_idx, tok['up'])])
                
    return matched 
    

############# Strategies that require special funcs #################

actually = lambda p, sent_idx: list(chain(extract_unigram_markers(p, sent_idx,UNIGRAM_MARKERS['Actually']), \
                               extract_dep_parse_markers(p, sent_idx, "the", parents=['point', 'reality', 'truth'], relation="det"), \
                               extract_dep_parse_markers(p, sent_idx, "fact", parents=['in'], relation="pobj")))
actually.__name__ = "Actually"
    

adv_just = lambda p, sent_idx: extract_dep_parse_markers(p, sent_idx, "just", relation="advmod")
adv_just.__name__ = "Adverb.Just"


apology = lambda p, sent_idx:list(chain(extract_unigram_markers(p, sent_idx, UNIGRAM_MARKERS['Apology']), \
                             extract_dep_parse_markers(p, sent_idx, "me", parents=['forgive', 'excuse'], relation='dobj'),\
                             extract_dep_parse_markers(p, sent_idx,'i', parents=['apologize'], relation="nsubj")))
apology.__name__ = "Apology"


gratitude = lambda p, sent_idx: list(chain(extract_unigram_markers(p, sent_idx, UNIGRAM_MARKERS["Gratitude"]),\
                                extract_dep_parse_markers(p, sent_idx, "i", parents=['appreciate']),\
                                extract_dep_parse_markers(p, sent_idx, "we", parents=['appreciate'])))
gratitude.__name__ = "Gratitude"


please = lambda p, sent_idx: [(tok, sent_idx, idx+1) for (tok, sent_idx, idx) in extract_unigram_markers(p[1:], sent_idx, UNIGRAM_MARKERS['Please'])]
please.__name__ = "Please"


swearing = lambda p, sent_idx: list(chain(extract_unigram_markers(p, sent_idx, UNIGRAM_MARKERS['Swearing']), \
                               extract_dep_parse_markers(p, sent_idx, "the", parents=['fuck', 'hell', 'heck'], relation="det")))
swearing.__name__ = "Swearing"


# Feature function list
MARKER_FNS = [actually, adv_just, apology, gratitude, please, swearing]
MARKER_FNS_NAMES = [fn.__name__ for fn in MARKER_FNS]


#### Extraction on parsed text

def extract_markers_from_sent(sent_parsed, sent_idx,\
                              unigram_markers= UNIGRAM_MARKERS,\
                              start_markers = START_MARKERS,\
                              ngram_markers = NGRAM_MARKERS,\
                              marker_fns = MARKER_FNS):
    
    
    sent_summary = defaultdict(list)
    
    # unigram
    for k, unigrams in unigram_markers.items():
        
        if k not in MARKER_FNS_NAMES:
            sent_summary[k].extend(extract_unigram_markers(sent_parsed, sent_idx, unigrams))

    # ngrams
    for k, ngrams in ngram_markers.items():
        
        if k not in MARKER_FNS_NAMES:
            sent_summary[k].extend(extract_ngram_markers(sent_parsed, sent_idx, ngrams))
    
    # starter
    for k in start_markers:
        if k not in MARKER_FNS_NAMES:
            sent_summary[k].extend(extract_starter_markers(sent_parsed, sent_idx,START_MARKERS[k]))
    
    
    # strategy by functions
    for fn in marker_fns:
        sent_summary.update({fn.__name__: fn(sent_parsed, sent_idx)})
    
    
    return sent_summary


def get_local_politeness_strategy_features(utt):
    
    markers = defaultdict(list)
    parsed = [x["toks"] for x in utt.meta["parsed"]]
    

    for sent_idx, sent_parsed in enumerate(parsed):

        sent_markers = extract_markers_from_sent(sent_parsed, sent_idx)
        # update markers 
        markers = {k:list(chain(markers[k], v)) for k,v in sent_markers.items()}
        
    
    features = {k: int(len(marker_list) > 0) for k, marker_list in markers.items()}
    
    return features, markers