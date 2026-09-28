"""Validate actual WAV decoding against lossless reference samples before retry."""
import json,subprocess
from pathlib import Path
import numpy as np
import soundfile as sf
import torch,torchaudio,torchcodec
from demucs.separate import load_track
root=Path('/root/Temporal_Resolution')
row=json.loads((root/'locked_split.json').read_text())['train'][0]
path=root/'audio'/(row['ID']+'.wav')
reference,sr=sf.read(path,dtype='float32',always_2d=True)
value,loaded_sr=torchaudio.load(str(path))
assert sr==loaded_sr==16000
np.testing.assert_allclose(value.numpy(),reference.T,atol=1e-7,rtol=0)
value=load_track(path,2,16000)
np.testing.assert_allclose(value.numpy(),np.repeat(reference.T,2,axis=0),atol=1e-7,rtol=0)
receipt=dict(audio_id=row['ID'],sample_tolerance=1e-7,torchcodec=torchcodec.__version__,
             torchaudio=torchaudio.__version__,ffmpeg=subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0])
(root/'decoder_verification.json').write_text(json.dumps(receipt,indent=2));print(receipt)
