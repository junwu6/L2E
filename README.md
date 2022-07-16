# Learning to Evolve: A Generic Meta-Learning Framework for Dynamic Transfer Learning

This is the official implementation of "Learning to Evolve: A Generic Meta-Learning Framework for Dynamic Transfer Learning". 


## Requirements
Python 3.6
numpy==1.18.1
torch==1.4.0
torchvision==0.5.0
higher==0.2.1
Pillow==7.0.0

## Training and testing
For dynamic transfer learning on a time evolving tasks on Office-31 or Image-CLEF, please run
python main.py

For dynamic transfer learning on a time evolving tasks on Caltran, please run
python utils/preprocess.py   ## pre-process Caltran to generate the time evolving tasks
python main_caltran.py

## Details
utils: load and save data
models: L2E as well as other baselines
