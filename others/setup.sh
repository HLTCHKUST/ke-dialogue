## SMD dataset
wget http://nlp.stanford.edu/projects/kvret/kvret_dataset_public.zip
mkdir SMD
unzip kvret_dataset_public.zip -d SMD

## M-WOZ 2.1
wget https://github.com/budzianowski/multiwoz/raw/master/data/MultiWOZ_2.1.zip
unzip MultiWOZ_2.1.zip

### Open DialKG
git clone https://github.com/facebookresearch/opendialkg.git
python split_opendialkg.py

### TASKMASTER
git clone https://github.com/google-research-datasets/Taskmaster.git

rm -rf .ipynb_checkpoints/
rm -rf MultiWOZ_2.1.zip
rm -rf __MACOSX/
rm -rf kvret_dataset_public.zip

### DialoGPT download
git clone https://github.com/circulosmeos/gdown.pl.git
cd gdown.pl
./gdown.pl https://drive.google.com/open?id=1ktfsH5iA5p-R4F2-qvbQhvh6g2D-SJ5T dialoGPT.zip
unzip dialoGPT.zip 
mv dialoGPT ../
cd .. 

## babi-data
wget https://github.com/facebookarchive/bAbI-tasks/archive/master.zip
mv master.zip babi.zip
unzip babi.zip