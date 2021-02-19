# *seq2att*: Learning, Visualizing and Exploring 16S rRNA Structure Using an Attention-based Deep Neural Network

*seq2att* is a command line interface to train a *Read2Pheno* model on customized 16S rRNA dataset.          
Drexel University EESI Lab, 2020        
Maintainer: Zhengqiao Zhao, zz374 at drexel dot edu        
Owner: Gail Rosen, gailr at ece dot drexel dot edu        

Conda installation:
```
conda create -n seq2att python=3.5.4 biopython=1.71 pandas=0.20.3 matplotlib=2.1.1 scipy=1.0.0 scikit-learn=0.19.1 tensorflow=1.9.0 keras=2.2.2
source activate seq2att
git clone https://github.com/z2e2/seq2att.git
cd seq2att
python setup.py install
```
