=====================================================================
# PartialSpoof Database v1.2
=====================================================================

Authors:
Lin Zhang (1,2), Xin Wang (1), Erica Cooper (1), Junichi Yamagishi (1,2), Jose Patino (3), Nicholas Evans (3)

Affiliations:
(1) National Institute of Informatics, Japan 
(2) The Graduate University for Advanced Studies, Japan
(3) Digital Security Department, EURECOM, France

National Institute of Informatics 
Copyright (c) 2021



Emails:
{zhanglin, wangxin, ecooper, jyamagis}@nii.ac.jp, {patino, evans}@eurecom.fr

* For the initial version of PartialSpoof v1.0
Arxiv: https://arxiv.org/abs/2104.02518
Samples: https://nii-yamagishilab.github.io/zlin-demo/IS2021/index.html
PartialSpoof Database v1.0: https://zenodo.org/record/4817532

* For the multi-task version of PartialSpoof v1.1
Arxiv: https://arxiv.org/abs/2107.14132
PartialSpoof Database v1.1 (including 0.16s segmental level labels): https://zenodo.org/record/5112031

* For fine-grained labels of PartialSpoof v1.2
Arxiv: http://arxiv.org/abs/2204.05177
PartialSpoof Database v1.2 (including segmental-level labels in different temporal resolutions and timestamp labels): This one.


===============================REFERENCES===============================

If you publish using any of the data in this dataset please cite the following papers:

#Initial version of PartialSpoof
@inproceedings{Zhang2021,
  archivePrefix = {arXiv},
  arxivId = {2104.02518},
  author = {Zhang, Lin and Wang, Xin and Cooper, Erica and Yamagishi, Junichi and Patino, Jose and Evans, Nicholas},
  eprint = {2104.02518},
  title = {{An Initial Investigation for Detecting Partially Spoofed Audio}},
  url = {http://arxiv.org/abs/2104.02518},
  booktitle = {Proc. Interspeech},
  year = {2021},
  pages={4264--4268},
  doi={10.21437/Interspeech.2021-738}
}

#ASVspoof2019 database
@article{WANG2020101114,
 author = {Wang, Xin and Yamagishi, Junichi and Todisco, Massimiliano and Delgado, H{\'{e}}ctor and Nautsch, Andreas and Evans, Nicholas and Sahidullah, Md and Vestman, Ville and Kinnunen, Tomi and Lee, Kong Aik and Juvela, Lauri and Alku, Paavo and Peng, Yu-Huai and Hwang, Hsin-Te and Tsao, Yu and Wang, Hsin-Min and Maguer, S{\'{e}}bastien Le and Becker, Markus and Henderson, Fergus and Clark, Rob and Zhang, Yu and Wang, Quan and Jia, Ye and Onuma, Kai and Mushika, Koji and Kaneda, Takashi and Jiang, Yuan and Liu, Li-Juan and Wu, Yi-Chiao and Huang, Wen-Chin and Toda, Tomoki and Tanaka, Kou and Kameoka, Hirokazu and Steiner, Ingmar and Matrouf, Driss and Bonastre, Jean-Fran{\c{c}}ois and Govender, Avashna and Ronanki, Srikanth and Zhang, Jing-Xuan and Ling, Zhen-Hua},
 doi = {https://doi.org/10.1016/j.csl.2020.101114},
 issn = {0885-2308},
 journal = {Computer Speech {\&} Language},
 keywords = {ASVspoof challenge,Anti-spoofing,Automatic speaker verification,Biometrics,Countermeasure,Media forensics,Presentation attack,Presentation attack detection,Replay,Text-to-speech synthesis,Voice conversion},
 pages = {101114},
 title = {{ASVspoof 2019: a large-scale public database of synthesized, converted and replayed speech}},
 url = {http://www.sciencedirect.com/science/article/pii/S0885230820300474},
 year = {2020}
}

#Multi-task learning for PartialSpoof
@inproceedings{Zhang2021PartialSpoofMTL,
  author={Lin Zhang and Xin Wang and Erica Cooper and Junichi Yamagishi},
  title={{Multi-task Learning in Utterance-level and Segmental-level Spoof Detection}},
  year=2021,
  booktitle={Proc. ASVspoof 2021 Workshop},
  pages={9--15},
  doi={10.21437/ASVSPOOF.2021-2},
  url = {http://arxiv.org/abs/2107.14132},
  archivePrefix = {arXiv},
  arxivId = {2107.14132}
}

#Summary and Multiple temporal resolution for PartialSpoof
@inproceedings{Zhang2022PSssl,
  author = {Zhang, Lin and Wang, Xin and Cooper, Erica and Evans, Nicholas and Yamagishi, Junichi},
  title = {{The PartialSpoof Database and Countermeasures for the Detection of Short Generated Audio Segments Embedded in a Speech Utterance}},
  eprint = {2204.05177},
  url = {http://arxiv.org/abs/2204.05177},
  archivePrefix = {arXiv},
  arxivId = {2204.05177}
}

======================== DATA STRUCTURE ==================================


1. Directory Structure
-------------------------------

./PartialSpoof_v1.2
├── database
│   ├── train:         data for the train set.
│   │   ├── con_wav:     audio files.
│   │   └── train.lst:   list of audio IDs.
│   ├── dev:           data for the development set.
│   │   ├── con_wav
│   │   └── dev.lst
│   ├── eval:          data for the evaluation set.
│   │   ├── con_wav
│   │   └── eval.lst
│   ├── protocols:     protocol files.
│   │   ├── ASVspoof2019_LA_asv_protocols
│   │   ├── ASVspoof2019_LA_asv_scores
│   │   ├── ASVspoof2019_LA_cm_protocols
│   │   ├── PartialSpoof_LA_asv_protocols
│   │   ├── PartialSpoof_LA_asv_scores
│   │   └── PartialSpoof_LA_cm_protocols
│   ├── vad:			 vad files with timestamp annotation.
│   │   ├── train
│   │   ├── dev
│   │   └── eval
│   └── segment_labels:  segment-level labels (with different temporal resolutions, ranging from 0.01 sec. to 0.64 sec.).
│       ├── train_seglab_*.**.npy
│       ├── dev_seglab_*.**.npy
│       └── eval_seglab_*.**.npy
└── README_v1.2


P.S. `database/{train, dev, eval, protocols}` for version 1.1 / 1.2 are the same as version 1.0. 
We only update the folder database/segment_labels and README_v1.1 for version 1.1 with segment-level labels in 0.16s.
and only update the folder database/{segment_labels, vad} and README_v1.2 for version 1.2 with segment-level labels in 0.01s..


2. Description for each files in detail.
---------------------------------------------------------

=== ./database/protocols/* ===
protocols contains ASV protocol files in ASCII format for t-DCF calculation of PartialSpoof countermeasure and official ASVspoof2019 countermeasure.

└── protocols
    ├── ASVspoof2019_LA_asv_protocols: ASV protocol files for ASVspoof2019
 	├── ASVspoof2019_LA_asv_scores: ASV score files for ASVspoof2019
    ├── ASVspoof2019_LA_cm_protocols:  CM protocol files for ASVspoof2019
    ├── PartialSpoof_LA_asv_protocols: ASV protocol files for PartialSpoof
 	├── PartialSpoof_LA_asv_scores: ASV score files for PartialSpoof
    └── PartialSpoof_LA_cm_protocols:  CM protocol files for PartialSpoof

  
    * ASV protocols:
    ├── ASVspoof2019_LA_asv_protocols: ASV protocol files for ASVspoof2019
    ├── PartialSpoof_LA_asv_protocols: ASV protocol files for PartialSpoof

	(We use the same structure for ASV protocol as ASVspoof2019. This intro is modified from README of ASVspoof2019)
	ASVspoof2019.LA.asv.<1>.<2>.<3>.txt and PartialSpoof.LA.asv.<1>.<2>.<3>.txt
        where
        <1> is either 'dev' or 'eval' based on whether the files describe the development or evaluation protocol,
        <2> ie either 'male (m)' or 'female (f)' separating the genders from each other or 'gender independent (gi)' 
            contains trials for both genders (male trials followed by female trials),
        <3> is either 'trl' or 'trn' (trl = trial list, trn = speaker enrollment list). 
            Since PartialSpoof use the same set of bona fide data from ASVspoof2019, trn files for PartialSpof is same as ASVspoof2019. 

        Trial (trl) file format for LA scenario:
        1st column: claimed speaker ID
        2nd column: test file ID
        3rd column: spoof attack ID (or 'bonafide' if the speech is not spoofed)
        4th column: key (target = target trial, nontarget = impostor trial, spoof = spoofing attack)

        Enrollment (trn) file format:
        1st column: ID of enrolled speaker
        2nd column: IDs of files used in the enrollment separated by commas

	* Baseline ASV scores:
	├── ASVspoof2019_LA_asv_scores: ASV scores files for ASVspoof2019, calculated by a baseline ASV system for t-DCF evaluation
	│	├── ASVspoof2019.LA.asv.dev.gi.trl.scores.txt
	│	└── ASVspoof2019.LA.asv.eval.gi.trl.scores.txt
	├── PartialSpoof_LA_asv_scores: ASV scores files for PartialSpoof, calculated by a baseline ASV system for t-DCF evaluation
		├── PartialSpoof.LA.asv.dev.gi.trl.scores.txt
		└── PartialSpoof.LA.asv.eval.gi.trl.scores.txt


	Since ASVspoof2019 and PartialSpoof have same bonafide trials, we use the same ASV model for both scenerios.
	(We use the same structure for ASV score with ASVspoof2019. This intro is modified from README of ASVspoof2019)
	
	ASVspoof2019.LA.asv.<1>.gi.trl.scores.txt and PartialSpoof.LA.asv.<1>.gi.trl.scores.txt are score files given by the ASV system,
		where <1> is either 'dev' or 'eval' based on whether the files describe the development or evaluation protocol.

		Each column is formatted as:
	    CM_KEY ASV_KEY SCORES

	        1) CM_KEY:     'bonafide' for genuine speech, or, the ID of the spoofing attack (A01 - A19, CON), 'CON' for PartialSpoof audio,
	        2) ASV_KEY:    'target' for claimed speaker, or, 'nontarget' for impostor speaker, or, 'spoof' for spoofing speech,
	        3) SCORES:     similarity score value.



    * CM protocols:
    ├── ASVspoof2019_LA_cm_protocols:  CM protocol files in ASCII format for ASVspoof2019 countermeasures. 
    │   ├──ASVspoof2019.LA.cm.train.trl.txt: training file list for ASVspoof2019
    │   ├──ASVspoof2019.LA.cm.dev.trl.txt: development trials for ASVspoof2019
    │   └──ASVspoof2019.LA.cm.eval.trl.txt: evaluation trials for ASVspoof2019
    └── PartialSpoof_LA_cm_protocols:  CM protocol files in ASCII format for PartialSpoof countermeasures. 
		├──PartialSpoof.LA.cm.train.trl.txt: training file list for PartialSpoof
		├──PartialSpoof.LA.cm.dev.trl.txt: development trials for PartialSpoof
		└──PartialSpoof.LA.cm.eval.trl.txt: evaluation trials for PartialSpoof

	(We use the same structure for CM protocol with ASVspoof2019. This intro is modified from README of ASVspoof2019)
	Each column of the protocol is formatted as:
	
	SPEAKER_ID AUDIO_FILE_NAME - SYSTEM_ID KEY
	
	     1) SPEAKER_ID:          LA_****, a 4-digit speaker ID
	     2) AUDIO_FILE_NAME:     For ASVspoof2019: LA_*_*******, name of the audio file
	                             For PartialSpoof: CON_*_*******, name of the audio file
	     3) -:                   This column is NOT used for LA.
	     4) SYSTEM_ID:           For ASVspoof2019: ID of the speech spoofing system (A01 - A19),  or, for bonafide speech SYSTEM-ID is left blank ('-')
	                             For PartialSpoof: ID of the speech spoofing system ('CON'),  or, for bonafide speech SYSTEM-ID is left blank ('-')
	     5) KEY:                 'bonafide' for genuine speech, or, 'spoof' for spoofing speech
	
	Note that: 
	
	     1) the third column is left blank (-) to make the structure coherent with physical access file list;
	     2) Brief description on LA spoofing systems, where TTS and VC denote text-to-speech and voice-conversion systems, CON denote PartialSpoof system:
	     
	     A01     TTS     neural waveform model
	     A02     TTS     vocoder
	     A03     TTS     vocoder
	     A04     TTS     waveform concatenation
	     A05     VC      vocoder
	     A06     VC      spectral filtering
	                     
	     A07     TTS     vocoder+GAN
	     A08     TTS     neural waveform
	     A09     TTS     vocoder
	     A10     TTS     neural waveform
	     A11     TTS     griffin lim
	     A12     TTS     neural waveform
	     A13     TTS_VC  waveform concatenation+waveform filtering
	     A14     TTS_VC  vocoder
	     A15     TTS_VC  neural waveform
	     A16     TTS     waveform concatenation
	     A17     VC      waveform filtering
	     A18     VC      vocoder
	     A19     VC      spectral filtering
	
	     CON     CON     PartialSpoof


=== ./database/{train,dev,eval}/con_wav ===
This directory contains the generated audio files.
	*.wav: wavform in Partialspoof database.
		LA_*.wav: original bonafide wav file after normalization (sv56), convert from *.flac from ASVspoof2019 database. 
		CON_*.wav: partialspoof wav file.


=== ./database/{train,dev,eval}/*.lst ===
	List of file IDs of {train, development, evaluation} set trials in this data set.


===./database/segment_labels/{train,dev,eval}_seglab_*.**.npy ===
This segment_labels folder contains the segment-level label in .npy format for each set. Labels are generated with frame length of *.**s and frame shift of *.**s.
The labels are: '0' for spoof, and '1' for bona fide.

Here is the commands to load numpy file of the labels with the temporal resolution as 0.16 second.
>>> import numpy as np
>>> train_seglab=np.load("train_seglab_0.16.npy", allow_pickle=True).item()
>>> train_seglab['CON_T_0000000']
array(['1', '1', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0',
       '0', '0', '0', '1', '1'], dtype='<U21')
>>> len(train_seglab)
25380


===NOTEs for segment-level labels.
Besides above *.npy files, segment-level labels at coarse temporal resolutions can be derived from *_seglab_0.01.npy by using following function:

```
def lab_in_scale(self, labvec, shift):
	""" Get segment-level labels in different temporal resolution based on the finer one.
    input:
    -------
      labvec:  List. segment-level label with the frame-shift as BASE_SHIFT.
       shift:  Int. Frame shift NUMBER of the output.

    output:
    -------
      new_lab: List<Int>, labels are generated from labvec, with the frame length of `BASE_SHIFT x shift`.
               '0' for spoof, and '1' for bona fide.

	"""
    num_frames = math.ceil(len(labvec) / shift)
    new_lab = np.zeros(num_frames, dtype=int)
    for idx in np.arange(num_frames):
        st, et  = int(idx * shift), int((idx+1)*shift)
        new_lab[idx] = min(labvec[st:et]) #only for binary. '0' for spoof

    return new_lab

```

===./database/vad/{train,dev,eval} ===
This vad folder contains the timestamp annotation for each set. 
For each <uttid>.vad file, the format of each line is: 
<start_time> <end_time> <label>

<start_time> and <end_time> are in second. 
<label> includes: '0' for spoof, '1' for bona fide, and '2' for non-speech

Here is an example file:
$ head -n 5 CON_T_0000000.vad                                           
0.0 0.5483125 2
0.5483125 1.141 0
1.141 1.1435625 2
1.1435625 1.5035625 0
1.5035625 1.5135625 2


============================== MD5SUM ================================
You can use `md5sum` to check the downloaded files:
$ md5sum ./*

ddd4cd3221b7210ac879f67452fb209e  ./database_dev.tar.gz
79c7c834d0d9979ecd374a98a059ea19  ./database_eval.tar.gz
699d81f020e4b7fa8f33747010e1cba8  ./database_protocols.tar.gz
c2bf6638e59ec7a5cf93c4a510fe4efe  ./database_segment_labels_v1.2.tar.gz
c4853ddd831e8e96b0e279fc0a512e7e  ./database_train.tar.gz
95e77e19a1bb4f2e79ed138fd35621ad  ./database_vad.tar.gz

============================== UPDATE ================================
20 May 2021, v1.0
Create the initial version 1.0 for PartialSpoof

-------------------
30 July 2021, v1.1
* Upload segment-level labels with frame_lenth=0.16s and frame_shift=0.16s.
* Update README_v1.1 with introduction for segment-level label.
* Update the link of multi-task learning paper.

-------------------
08 Dec. 2021, v1.2
* Upload segment-level labels with frame_lenth=0.01s and frame_shift=0.01s.
* Upload vad (timestamp) files for all subsets.
* Update README_v1.2 with (1) introduction for segment-level label with different temporal resolutions,
						  (2) function to derive segment-level labels with different temporal resolutions,
						  (3) and introduction of vad files with timestamp annotations.


============================== COPYING ================================

This repository is made available under Creative Commons Attribution License (CC-BY). 

Regarding Creative Commons License: Attribution 4.0 International (CC BY 4.0), 
please see https://creativecommons.org/licenses/by/4.0/

THIS DATABASE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND 
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, 
OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
ARISING IN ANY WAY OUT OF THE USE OF THIS DATABASE, EVEN IF ADVISED OF THE 
POSSIBILITY OF SUCH DAMAGE




====================== ACKNOWLEDGEMENTS ================================

Thanks Dr. Ville Vestman from University of Eastern Finland for sharing an ASV model for min-tDCF evaluation in the initial paper. This study was partially supported by the Japanese-French joint national VoicePersonae project supported by JST CREST (JPMJCR18A6 and JPMJCR20D3) and the ANR (ANR-18-JSTS-0001), MEXT KAKENHI Grants (16H06302, 18H04120, 18H04112, 18KT0051, 21H04906, 21K17775, 21K11951), Japan, and Google AI for Japan program.

This database is based on the ASVspoof 2019 database and this database is distributed under the Open Data Commons Attribution License (ODC-By). 
Regarding Open Data Commons Attribution License (ODC-By), please see https://opendatacommons.org/licenses/by/1.0/index.html
You can download ASVspoof 2019 from https://datashare.ed.ac.uk/handle/10283/3336

