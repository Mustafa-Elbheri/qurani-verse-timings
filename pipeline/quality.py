"""Fail-closed, evidence-bound acceptance of automatically aligned Quran units."""
import hashlib
import json
import math
from pathlib import Path

from quran_ref import normalize

POLICY_VERSION = 3
MIN_UNIT_MS = 120
MAX_BOUNDARY_DRIFT_MS = 80


def reference_digest(ayahs):
    raw = json.dumps(ayahs, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(raw).hexdigest()


def compact(text):
    return ''.join(normalize(text).split())


def assess(ayah, observations, source_sha256, reference_sha256):
    """Recompute acceptance from raw word observations, never a saved PASS flag."""
    reasons = []
    if len(observations) < 2:
        return ['two_independent_windows_required']
    windows = {(o.get('window_start_ms'), o.get('window_end_ms')) for o in observations}
    if len(windows) < 2:
        reasons.append('duplicate_observation_window')
    bounds = []
    expected = compact(ayah.get('asr_text', ayah['text']))
    for obs in observations:
        if obs.get('model_name') not in ('small', 'large-v3', 'wav2vec2-native'):
            reasons.append('unrecognized_evidence_model')
        if obs.get('source_sha256') != source_sha256:
            reasons.append('evidence_source_mismatch')
        if obs.get('reference_sha256') != reference_sha256:
            reasons.append('evidence_reference_mismatch')
        model_hash = obs.get('model_sha256', '')
        if len(model_hash) != 64 or any(c not in '0123456789abcdef' for c in model_hash):
            reasons.append('model_identity_missing')
        words = obs.get('words', [])
        if not words:
            reasons.append('no_acoustic_words')
            continue
        if compact(''.join(w['word'] for w in words)) != expected:
            reasons.append('reference_word_coverage_mismatch')
        probs = [w.get('probability') for w in words]
        if any(not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1 for p in probs):
            reasons.append('word_probability_missing')
        elif min(probs) < .15 or sum(probs) / len(probs) < .75 or min(probs[0], probs[-1]) < .5:
            reasons.append('insufficient_acoustic_support')
        previous = -1
        for word in words:
            start, end = word['start_ms'], word['end_ms']
            if end - start < 20 or start < previous:
                reasons.append('instant_or_overlapping_word')
            if start < obs['window_start_ms'] or end > obs['window_end_ms'] + 1:
                reasons.append('word_outside_source_window')
            previous = end
        start, end = words[0]['start_ms'], words[-1]['end_ms']
        if end - start < MIN_UNIT_MS:
            reasons.append('implausible_unit_duration')
        bounds.append((start, end))
        if obs.get('model_name') == 'large-v3':
            from rapidfuzz.fuzz import ratio
            free = ''.join(w['word'] for w in obs.get('free_words', []))
            if not obs.get('free_identity_verified') or ratio(expected, compact(free)) < 92:
                reasons.append('large_free_identity_not_verified')
        if obs.get('model_name') == 'wav2vec2-native':
            from rapidfuzz.fuzz import ratio
            if ratio(expected, compact(obs.get('free_ctc',''))) < 92:
                reasons.append('ctc_free_identity_not_verified')
            chars = obs.get('char_evidence', [])
            if not chars or min(c['probability'] for c in chars[:3]+chars[-3:]) < .5:
                reasons.append('ctc_native_edge_not_supported')
            native = obs.get('ctc_words', [])
            timing = obs.get('timing_words', [])
            if not native or len(timing) != len(words) or len(native) != len(words):
                reasons.append('independent_whisper_bounds_missing')
            else:
                if abs(native[0]['start_ms']-timing[0]['start_ms']) > 80 or abs(native[-1]['end_ms']-timing[-1]['end_ms']) > 80:
                    reasons.append('ctc_whisper_boundary_disagreement')
                if any((a['start_ms'],a['end_ms']) != (b['start_ms'],b['end_ms']) for a,b in zip(words,timing)):
                    reasons.append('unsupported_combined_word_bounds')
    if len(bounds) >= 2:
        if max(b[0] for b in bounds) - min(b[0] for b in bounds) > MAX_BOUNDARY_DRIFT_MS:
            reasons.append('unstable_onset')
        if max(b[1] for b in bounds) - min(b[1] for b in bounds) > MAX_BOUNDARY_DRIFT_MS:
            reasons.append('unstable_endpoint')
    return sorted(set(reasons))


def audit_review(proposal, review, directory, ayahs):
    errors = []
    directory = Path(directory).resolve()
    expected_ref = reference_digest(ayahs)
    source = proposal.get('source_sha256')
    if proposal.get('approved') is not True:
        errors.append('proposal_not_automatically_approved')
    if review.get('source_sha256') != source:
        errors.append('review_source_mismatch')
    if review.get('reference_sha256') != expected_ref:
        errors.append('review_reference_mismatch')
    if review.get('policy_version') != POLICY_VERSION:
        errors.append('missing_current_acceptance_policy')
    if review.get('pending_cases'):
        errors.append('unresolved_cases')
    if review.get('first_and_last_checked') is not True:
        errors.append('first_or_last_not_verified')
    audio = list(directory.glob('audio.*'))
    audio = [p for p in audio if p.suffix.lower() in ('.mp3', '.wav', '.m4a', '.ogg', '.flac')]
    if len(audio) != 1:
        errors.append('exactly_one_native_source_required')
    else:
        from runtime import digest
        if digest(audio[0]) != source:
            errors.append('native_source_hash_mismatch')
    by_num = {v['num']: v for v in proposal.get('verses', [])}
    proofs = review.get('unit_evidence', {})
    for num, ayah in enumerate(ayahs, 1):
        descriptors = proofs.get(str(num), [])
        observations = []
        for descriptor in descriptors:
            try:
                path = (directory / descriptor['file']).resolve()
                if not path.is_relative_to(directory):
                    raise ValueError('escaping evidence path')
                raw = path.read_bytes()
                if hashlib.sha256(raw).hexdigest() != descriptor['sha256']:
                    raise ValueError('evidence hash mismatch')
                document = json.loads(raw)
                observation = document['units'][str(num)]
                observations.append({**document['identity'], **observation})
            except (KeyError, ValueError, OSError):
                errors.append(f'unit-{num}:missing_or_changed_evidence')
        for reason in assess(ayah, observations, source, expected_ref):
            errors.append(f'unit-{num}:{reason}')
        if num in by_num and observations:
            verse = by_num[num]
            starts = [o['words'][0]['start_ms'] for o in observations if o.get('words')]
            ends = [o['words'][-1]['end_ms'] for o in observations if o.get('words')]
            if starts and abs(verse['start_ms'] - sum(starts) / len(starts)) > 110:
                errors.append(f'unit-{num}:exported_onset_not_supported')
            if ends and abs(verse['end_ms'] - sum(ends) / len(ends)) > 180:
                errors.append(f'unit-{num}:exported_endpoint_not_supported')
    return errors
