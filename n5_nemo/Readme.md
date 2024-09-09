-------------------------------------------------------------------------------
                         Samrómur NeMo Recipe 22.06
-------------------------------------------------------------------------------

Authors               : Carlos Daniel Hernández Mena (carlosm@ru.is)

Programming Languages : NVIDIA-NeMo, Python3, Bash

Recommended use       : speech recognition.

-------------------------------------------------------------------------------
Description
-------------------------------------------------------------------------------

The "Samrómur NeMo Recipe 22.06" is a code recipe intended to show how to 
integrate the corpus "Samromur 21.05" [1] and the "6-GRAM Language Model in 
Icelandic for NeMo (Binary Format) 22.06" [2] to create automatic speech 
recognition systems using the NVIDIA-NeMo framework [3].

This recipe was created by the Language and Voice Lab (LVL) at Reykjavík
University during 2022 and it is also available in GitHub at the
following link:

   https://github.com/cadia-lvl/samromur-asr/tree/n5_samromur/n5_samromur

In order to set the scripts up, it is necessary to install minimum 
requirements and to specify some paths; all of them are indicated in the 
"run.sh" script of each recipe.

-------------------------------------------------------------------------------
Citation
-------------------------------------------------------------------------------

When publishing results based on the models please refer to:

   Mena, Carlos; "Samrómur NeMo Recipe 22.06". Web Download. 
   Reykjavik University: Language and Voice Lab, 2022.

Contact: Carlos Mena (carlosm@ru.is)

License: CC BY 4.0

-------------------------------------------------------------------------------
Installation
-------------------------------------------------------------------------------

- You have to install in your system:

	* sox
	* libsndfile1
	* ffmpeg

- Create the conda environment

	$ conda create -n nemo_r150 python=3.7 anaconda

- Activate the environment

	$ conda activate nemo_r150

- Install Pytorch

	$ conda install pytorch torchvision torchaudio cudatoolkit -c pytorch

- Install NVIVIDIA-APEX
	$ git clone https://github.com/NVIDIA/apex
	$ cd apex
	$ pip install -v --disable-pip-version-check --no-cache-dir ./
	
- Install some packages
	$ pip install wget
	$ pip install unidecode

- Install NeMo

	$ pip install git+https://github.com/NVIDIA/NeMo.git@r1.5.0#egg=nemo_toolkit[all]
	
- Install LM Decoders

	$ git clone https://github.com/NVIDIA/NeMo.git
	$ cd NeMo/scripts/asr_language_modeling/ngram_lm
	$ bash install_beamsearch_decoders.sh
	
- Test Instalation

	$ python
	$ import nemo
	$ import nemo.collections.asr as nemo_asr

-------------------------------------------------------------------------------
Acknowledgements
-------------------------------------------------------------------------------

This initiative was funded by the Language Technology Programme for Icelandic 
2019-2023. The programme, which is managed and coordinated by Almannarómur, 
is funded by the Icelandic Ministry of Education, Science and Culture.

-------------------------------------------------------------------------------
References
-------------------------------------------------------------------------------

[1] Mollberg, David Erik ; et al; Samromur 21.05. Web Downloading: 
    http://hdl.handle.net/20.500.12537/185

[2] Mena, Carlos; et al; 6-GRAM Language Model in Icelandic for NeMo (Binary 
    Format) 22.06. Web Downloading: http://hdl.handle.net/20.500.12537/226

[3] Kuchaiev, O., Li, J., Nguyen, H., Hrinchuk, O., Leary, R., Ginsburg, 
    B., ... & Cohen, J. M. (2019). Nemo: a toolkit for building ai 
    applications using neural modules. arXiv preprint arXiv:1909.09577.

-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

