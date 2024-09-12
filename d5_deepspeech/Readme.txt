# Samrómur DeepSpeech Recipe 22.06

## Description

This is a code recipe to create an ASR system based on the ASR corpus [Samrómur 21.05](http://hdl.handle.net/20.500.12537/189) and the [DeepSpeech Scorer for Icelandic 22.06](http://hdl.handle.net/20.500.12537/227) using the Mozilla's [DeepSpeech recognizer](https://github.com/mozilla/DeepSpeech).

This recipe was created by the Language and Voice Lab (LVL) at Reykjavík
University during 2022 and it is also available in GitHub at the
following link:

   https://github.com/cadia-lvl/samromur-asr/tree/d5_samromur/d5_samromur

In order to set the scripts up, it is necessary to install minimum 
requirements and to specify some paths; all of them are indicated in the 
"run.sh" script of each recipe.

-------------------------------------------------------------------------------
Citation
-------------------------------------------------------------------------------

When publishing results based on the models please refer to:

   Mena, Carlos; "Samrómur DeepSpeech Recipe 22.06". Web Download. 
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

- Create the Conda environment for DeepSpeech.

	$ conda create -n deepspeech python=3.7 anaconda
	$ conda activate deepspeech
	$ conda install cudatoolkit=10.1
	$ conda install numpy=1.18 scipy
	$ conda install -c conda-forge git-lfs
	$ conda install -c conda-forge nvidia-apex
	$ conda install pycodestyle=2.7.0 pyflakes=2.3.1
	$ conda install lxml requests python-dateutil pytz

- Intall DeepSpeech

	$ pip install deepspeech-gpu==0.10.0-alpha.3 ds-ctcdecoder==0.10.0-alpha.3
	$ pip install tensorflow-gpu==1.15.4
	$ git clone https://github.com/mozilla/DeepSpeech.git
	$ cd DeepSpeech
	$ python setup.py install

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

[2] Mena, Carlos; et al; DeepSpeech Scorer for Icelandic 22.06. 
    Web Downloading: http://hdl.handle.net/20.500.12537/227

[3] Amodei, D., Ananthanarayanan, S., Anubhai, R., Bai, J., Battenberg, 
    E., Case, C., ... & Zhu, Z. (2016, June). Deep speech 2: End-to-end 
    speech recognition in english and mandarin. In International conference 
    on machine learning (pp. 173-182). PMLR.

-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

