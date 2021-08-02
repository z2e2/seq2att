# *seq2att*: Learning, Visualizing and Exploring 16S rRNA Structure Using an Attention-based Deep Neural Network

*seq2att* is a command line interface to train a *Read2Pheno* model on customized 16S rRNA dataset.          
Drexel University EESI Lab, 2020        
Maintainer: Zhengqiao Zhao, zz374 at drexel dot edu        
Owner: Gail Rosen, gailr at ece dot drexel dot edu        

## Conda installation:
```
## If you don't have compatible GPUs for training (CPU only) 
conda create -n seq2att python=3.5.4 biopython=1.71 pandas=0.20.3 matplotlib=2.1.1 scipy=1.0.0 scikit-learn=0.19.1 tensorflow=1.9.0 keras=2.2.2
source activate seq2att
# If you want to use GPUs to accelerate your training process
conda create -n seq2att python=3.5.4 biopython=1.71 pandas=0.20.3 matplotlib=2.1.1 scipy=1.0.0 scikit-learn=0.19.1 tensorflow-gpu=1.9.0 keras-gpu=2.2.2
source activate seq2att

git clone https://github.com/z2e2/seq2att.git
cd seq2att
python setup.py install
```

## Steps to use this tool:
#### 1. Prepare your data
A raw data directory is required. It contains a comma separated file (CSV) and FASTA nucleic acid files for different samples. The directory structure of raw data is shown below:
```
raw_data
│   meta_data.csv  
|   SAMPLE_1.fna
|   SAMPLE_2.fna
|   SAMPLE_3.fna
|   ...
```
Formats of the required files is shown below:
1. *meta_data.csv* format example:

| sample_id | label  |
|-----------|--------|
| SAMPLE_1  | feces  |
| SAMPLE_2  | tongue |
| SAMPLE_3  | feces  |
| SAMPLE_4  | skin   |
| ...       | ...    |

2. SAMPLE_1.fna format example:
```
>SAMPLE_1_1
GCGAGCGAAGTTCGGAATTACTGGGCGTAAAGGGTGTGTA
>SAMPLE_1_2
GCGAGCGTTGTTCGGAACCACTGGGCGTAAAGGGTGTGTA
>SAMPLE_1_3
GCGAGCGTTGTTCGGAATTACTGGGCGTAGAGGGTGTGTA
```
#### 2. Edit the configuration file
The user should edit the `config.yml` [file](https://github.com/EESI/seq2att/blob/master/config.yml) to configure the model specs including the path to your data and the hyperparameters of the model. The following parameters should be edited based on your data. Please also carefully review the `config.yml` file's comments for more detailed information for the rest of parameters and make sure you also modified them accordingly.
1. `in_dir`: the absolute path to the raw data directory (should be created by the user in advance).
2. `out_dir`: the absolute path to the processed data directory (should be created by the user in advance and files will be generated by this program).
3. `num_train_samples_per_cls`: the number of training sample per class. Note that the rest of samples will be automatically placed in the testing set used for evaluation. If you decide to use all the data for training, then the testing set will be empty and then hence the downstreaming evaluation command won't work.
4. `SEQLEN`: sequence length.
5. `BASENUM`: the number of unique characters (nucleotides/amino acids) (e.g., for DNA reads, there are 4 major bases for DNA sequence input, therefore you can set `BASENUM` to 4).
6, `Ty`: the number of target classes (`Ty` >= 2).
7. `save_model_path`: the absolute path to the saved model directory (should be created by the user in advance).
8. `n_workers`: this is used for a Keras function. It defines the number of threads generating batches in parallel. According to Kera, batches are computed in parallel on the CPU and passed on the fly onto the GPU for neural network computations.

#### 3. Use the default command to run the end-to-end model 
The following command trains and evaluates the end-to-end model based on the config file you just edited. To be specific, this command will preprocess the data to convert them into pickle files, then train the model specificed in the `config.yml` file and finally evaluate the model on testing data. 
```
seq2att default -m config.yml
```
Note that the `-m` flag passes the path to the config.yml file to the program.           
If you prefer to run those steps one by one so that you have more control over the process, the following commands will produce the same outcome as the default command shown above.
```
seq2att build -m config.yml
seq2att train -m config.yml
```

#### 4. Visualize the read embedding and attention weights
We recommend you to use our [python package](https://github.com/EESI/sequence_attention/tree/master) to visualize the attention weights because you can have more control over the figure generation. For a quick and dirty visualization in commandline, please use the following command:
```
seq2att visualize -m config.yml -data datafile -taxa taxadata -name taxaname
```
This command does need the user to prepare additional files. First, `datafile` is a pickle file that contains `X_visual` (N by SEQ_LEN by NUMBASE) in *numpy array* and `y_visual` (phenotypic labels in integers) in *numpy array*,  `taxadata` a list of taxonomic labels of those sequences (e.g., genus level labels). `taxaname` is the name of the taxon of interest.
