"""Shared model instances and immutable identities; no silent CPU fallback."""
import ctypes
import hashlib
import os
from pathlib import Path
import threading

_models = {}
_identities = {}
_lock = threading.Lock()


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _import_backend():
    # Reuse PyTorch's CUDA DLL family on Windows. Loading another cublas copy
    # deadlocks this installed environment. Do not modify or copy system DLLs.
    if os.name != 'nt':
        import faster_whisper
        return faster_whisper
    import torch
    original = ctypes.CDLL
    torch_lib = Path(torch.__file__).parent / 'lib'

    class CanonicalDLL(original):
        def __init__(self, name, *args, **kwargs):
            candidate = torch_lib / Path(str(name)).name
            if 'site-packages' in str(name) and 'nvidia' in str(name) and candidate.is_file():
                name = str(candidate)
            super().__init__(name, *args, **kwargs)

    ctypes.CDLL = CanonicalDLL
    try:
        import faster_whisper
        return faster_whisper
    finally:
        ctypes.CDLL = original


def model_directory(name):
    configured = os.environ.get('QURANI_MODELS_DIR')
    roots = [Path(configured)] if configured else []
    if os.name == 'nt':
        roots.append(Path(os.environ['LOCALAPPDATA']) / 'Programs' / 'MutunTranscription' / 'models')
    for root in roots:
        path = root / ('faster-whisper-' + name)
        if (path / 'model.bin').is_file():
            return path
    from faster_whisper.utils import download_model
    return Path(download_model(name))


def whisper(name, device='cpu', compute_type='int8'):
    key = (name, device, compute_type)
    with _lock:
        if key not in _models:
            _import_backend()
            import stable_whisper
            directory = model_directory(name)
            _models[key] = stable_whisper.load_faster_whisper(
                str(directory), device=device, compute_type=compute_type,
                local_files_only=True, num_workers=1,
                cpu_threads=int(os.environ.get('QURANI_CPU_THREADS', '4')))
            _identities[key] = digest(directory / 'model.bin')
        return _models[key], _identities[key]
