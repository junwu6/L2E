# L2E
An implementation for "A Unified Meta-Learning Framework for Dynamic Transfer Learning" (IJCAI'22) [[Paper]](https://www.ijcai.org/proceedings/2022/496)[[arXiv]](https://arxiv.org/pdf/2207.01784.pdf).

## Environment Requirements
The code has been tested under Python 3.6.5. The required packages are as follows:
* numpy==1.18.1
* torch==1.4.0
* torchvision==0.5.0
* higher==0.2.1
* Pillow==7.0.0

## Data Sets
We used the following data sets in our experiments:
* [Office-31](https://people.eecs.berkeley.edu/~jhoffman/domainadapt/)
* [Image-CLEF](https://drive.google.com/file/d/0B9kJH0-rJ2uRS3JILThaQXJhQlk/view)
* [Caltran](http://cma.berkeleyvision.org/)

## Run the Codes
For dynamic transfer learning on a time evolving tasks on Office-31 or Image-CLEF, please run
```
python main.py
```

For dynamic transfer learning on a time evolving tasks on Caltran, please run
```
python utils/preprocess.py   ## pre-process Caltran to generate the time evolving tasks
python main_caltran.py
```

## Acknowledgement
This is the latest source code of **L2E** for IJCAI-2022. If you find that it is helpful for your research, please consider to cite our paper:

```
@inproceedings{wu2022unified,
  title={A Unified Meta-Learning Framework for Dynamic Transfer Learning},
  author={Wu, Jun and He, Jingrui},
  booktitle={Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence},
  pages={3573--3579},
  year={2022}
}
```
