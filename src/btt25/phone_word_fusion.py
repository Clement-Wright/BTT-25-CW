"""Score a fixed union of native and single-phone-edit word proposals."""
import numpy as np

from btt25.ctc import ctc_minimum_frames, greedy_ctc
from btt25.phone_edit_candidates import proposals
from btt25.phone_robustness import path_scores


def viterbi_scores(logits, paths):
    """Best legal CTC path score for each fixed pronunciation, with blank ID zero."""
    probabilities = logits.astype(np.float64)
    probabilities -= np.logaddexp.reduce(probabilities,axis=1)[:,None]
    n = len(paths)
    lengths = np.asarray([2*len(p)+1 for p in paths])
    width = int(lengths.max())
    labels = np.zeros((n,width),dtype=np.intp)
    for i,p in enumerate(paths):
        if any(t<=0 or t>=logits.shape[1] for t in p):
            raise ValueError("Unexpected blank or unknown phone in proposed pronunciation")
        labels[i,1:lengths[i]:2] = p
    skip = np.zeros_like(labels,dtype=bool)
    skip[:,2:] = (labels[:,2:]!=0) & (labels[:,2:]!=labels[:,:-2])
    state = np.full((n,width),-np.inf)
    state[:,0] = 0.
    for frame in probabilities:
        new = state.copy()
        new[:,1:] = np.maximum(new[:,1:],state[:,:-1])
        new[:,2:] = np.maximum(new[:,2:],np.where(skip[:,2:],state[:,:-2],-np.inf))
        state = new + frame[labels]
    final = np.maximum(state[np.arange(n),lengths-1],state[np.arange(n),np.maximum(lengths-2,0)])
    if not np.isfinite(final).all():
        raise ValueError("Infeasible proposal pronunciation")
    return final


def assemble(native, logits, lex, lm, edit_config, registration, *, archived=False):
    """Return ordered source pools. Each text stores its path and summed acoustic scores."""
    top = native[:registration["native_limit"]]
    if archived:
        orig = {c["text"]:(c["path_acoustic"],c["ctc_log_mass"]) for c in top}
    else:
        sums = path_scores(logits,[c["phoneme_ids"] for c in top]) if top else []
        orig = {c["text"]:(c["component_scores"]["acoustic"],float(val)) for c,val in zip(top,sums,strict=True)}
    phones = greedy_ctc(logits.argmax(1).tolist())
    generated = [(t,p) for t,p in proposals(phones,lex,lm,edit_config)
                 if ctc_minimum_frames(p)<=len(logits)][:registration["proposal_limit"]]
    combined = dict(orig)
    if generated:
        paths = [p for t,p in generated]
        peak, mass = viterbi_scores(logits,paths),path_scores(logits,paths)
        for (t,p),v,s in zip(generated,peak,mass,strict=True):
            a,b = combined.get(t,(-np.inf,-np.inf))
            combined[t] = (max(a,float(v)),max(b,float(s)))
    return orig, combined, generated


def select(pool, scores, weight, aligned):
    acoustic_index = 1 if aligned else 0
    return max(pool,key=lambda t:pool[t][acoustic_index] + weight*scores[t],default="")
