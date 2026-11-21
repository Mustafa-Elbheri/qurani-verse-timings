"""Native-source Arabic CTC evidence, used only after Small and Large fail."""
import os
from pathlib import Path
import numpy as np

from quran_ref import normalize
from runtime import digest

_loaded = {}


def backend(device):
    if device in _loaded:
        return _loaded[device]
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    base = os.environ.get('QURANI_MODELS_DIR')
    if not base and os.name == 'nt':
        base = str(Path(os.environ['LOCALAPPDATA'])/'Programs'/'MutunTranscription'/'models')
    location = Path(base)/'wav2vec2-arabic' if base else None
    if location and location.is_dir():
        name = str(location)
    else:
        from huggingface_hub import snapshot_download
        name = snapshot_download('jonatasgrosman/wav2vec2-large-xlsr-53-arabic')
        location = Path(name)
    processor = Wav2Vec2Processor.from_pretrained(name)
    model = Wav2Vec2ForCTC.from_pretrained(name).to(device).eval()
    if device == 'cuda':
        model.half()
    weights = list(location.glob('*.safetensors')) or list(location.glob('pytorch_model*.bin'))
    if len(weights) != 1:
        raise RuntimeError('CTC requires an unambiguous weight identity')
    identity = digest(weights[0])
    _loaded[device] = processor, model, identity
    return _loaded[device]


def ctc_path(log_probs, labels, blank=0):
    """Vectorized CTC Viterbi; target characters may never be skipped."""
    frames = len(log_probs)
    if not labels or frames < len(labels):
        raise ValueError('No supported CTC path')
    expanded = np.full(2*len(labels)+1, blank, dtype=np.int32)
    expanded[1::2] = labels
    states = len(expanded)
    previous = np.full(states,-np.inf,dtype=np.float32)
    previous[0] = log_probs[0,blank]
    previous[1] = log_probs[0,labels[0]]
    skip = np.zeros(states,dtype=bool)
    skip[2:] = (expanded[2:]!=blank)&(expanded[2:]!=expanded[:-2])
    back = np.zeros((frames,states),dtype=np.uint8)
    for t in range(1,frames):
        a=previous
        b=np.concatenate(([-np.inf],previous[:-1])).astype(np.float32)
        c=np.concatenate(([-np.inf,-np.inf],previous[:-2])).astype(np.float32)
        c[~skip]=-np.inf
        candidates=np.stack((a,b,c))
        choice=np.argmax(candidates,axis=0)
        back[t]=choice
        previous=np.max(candidates,axis=0)+log_probs[t,expanded]
    state=states-1 if previous[-1]>=previous[-2] else states-2
    if not np.isfinite(previous[state]):
        raise ValueError('No complete CTC path')
    trace=np.empty(frames,dtype=np.int32)
    for t in range(frames-1,-1,-1):
        trace[t]=state
        if t:state-=int(back[t,state])
    spans=[]
    for i,label in enumerate(labels):
        assigned=np.flatnonzero(trace==2*i+1)
        if not len(assigned):raise ValueError('Missing CTC character')
        spans.append((int(assigned[0]),int(assigned[-1])+1,float(np.exp(log_probs[assigned,label]).max())))
    return spans


def observe(audio, ayahs, numbers, begin_ms, end_ms, target, device, source_hash, reference_hash):
    import torch
    processor,model,model_hash=backend(device)
    clip=audio[begin_ms*16:end_ms*16].copy()
    if not len(clip) or end_ms-begin_ms>120000:raise ValueError('Invalid native CTC window')
    inputs=processor(clip,sampling_rate=16000,return_tensors='pt')
    values=inputs.input_values.to(device=device,dtype=next(model.parameters()).dtype)
    with torch.inference_mode():
        probabilities=model(values).logits.float().softmax(-1)[0].cpu().numpy()
    vocab=processor.tokenizer.get_vocab()
    pad=processor.tokenizer.pad_token_id
    delimiter=processor.tokenizer.word_delimiter_token
    folded={'<blank>':[pad],'|':[]}
    for token,index in vocab.items():
        if index==pad:continue
        char='|' if token==delimiter else normalize(token)
        if len(char)==1:folded.setdefault(char,[]).append(index)
    folded={k:v for k,v in folded.items() if v}
    characters=list(folded)
    indices={c:i for i,c in enumerate(characters)}
    p=np.stack([probabilities[:,ids].sum(axis=1) for ids in folded.values()],axis=1)
    emissions=np.log(np.maximum(p,1e-30))
    labels=[];owners=[];words=[]
    for number in numbers:
        for word in ayahs[number-1]['asr_text'].split():
            if labels:
                labels.append(indices['|']);owners.append(None)
            wi=len(words);words.append((number,word))
            for char in normalize(word):
                if char not in indices:raise ValueError('Reference character missing from CTC vocabulary')
                labels.append(indices[char]);owners.append(wi)
    spans=ctc_path(emissions,labels,indices['<blank>'])
    step=(end_ms-begin_ms)/len(p)
    aligned=[];char_evidence=[]
    for wi,(number,word) in enumerate(words):
        selected=[s for owner,s in zip(owners,spans) if owner==wi]
        if not selected:continue
        start=round(begin_ms+selected[0][0]*step)
        end=round(begin_ms+selected[-1][1]*step)
        if number==target:
            aligned.append(dict(word=word,start_ms=start,end_ms=end,probability=sum(s[2] for s in selected)/len(selected)))
            char_evidence.extend(dict(start_ms=round(begin_ms+s[0]*step),end_ms=round(begin_ms+s[1]*step),probability=s[2]) for s in selected)
    if not aligned:raise ValueError('Target has no supported native CTC words')
    lo=max(0,round((aligned[0]['start_ms']-begin_ms)/step))
    hi=min(len(p),round((aligned[-1]['end_ms']-begin_ms)/step)+1)
    ids=np.argmax(p[lo:hi],axis=1)
    free=[];previous=None
    for index in ids:
        if int(index)!=previous and int(index)!=indices['<blank>']:free.append(characters[int(index)])
        previous=int(index)
    from rapidfuzz.fuzz import ratio
    expected=''.join(normalize(ayahs[target-1]['asr_text']).split())
    free_text=''.join(free).replace('|','')
    identity_score=ratio(expected,free_text)
    ends=char_evidence[:3]+char_evidence[-3:]
    return dict(source_sha256=source_hash,reference_sha256=reference_hash,model_sha256=model_hash,
                model_name='wav2vec2-native',window_start_ms=begin_ms,window_end_ms=end_ms,
                words=aligned,char_evidence=char_evidence,free_ctc=free_text,
                free_identity_score=identity_score,free_identity_verified=identity_score>=92,
                native_boundary_support=bool(ends) and min(c['probability'] for c in ends)>=.5)
