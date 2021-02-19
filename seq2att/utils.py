import os
import random
import pandas as pd
import pickle
import logging

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)

def fna_to_dict(sample_id, label, in_dir, out_dir):
    '''
    convert fna file to dictionary
    '''
    
    out_name = '{}/{}/{}.pkl'.format(out_dir, label, sample_id)
    filename = '{}/{}.fna'.format(in_dir, sample_id)
    
    meta_data_read_list = []
    read = ''
    f_sample = open(filename)
    read_dict = {}
    for line in f_sample:
        if line[0] == '>':
            if len(read) != 0:
                read_dict[header] = read
                meta_data_read_list.append(header)
                read = ''
            header = line[1:].strip()
        else:
            read += line.strip()
    if len(read) != 0:
        read_dict[header] = read
        meta_data_read_list.append(header)
    f_sample.close()
    pickle.dump(read_dict, open(out_name, 'wb'))
    return meta_data_read_list

def split_sample(sample_id, label, in_dir, out_dir):
    '''
    split reads in a sample into seperate files
    '''
    
    file_dir = '{}/{}/{}'.format(out_dir, label, sample_id)
    filename = '{}/{}.fna'.format(in_dir, sample_id)
    
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    
    meta_data_read_list = []
    read = ''
    f_sample = open(filename)
    for line in f_sample:
        if line[0] == '>':
            if len(read) != 0:
                with open('{}/{}/{}/{}.fna'.format(out_dir, label, sample_id, header), 'w+') as f_read:
                    f_read.write(read)
                    meta_data_read_list.append(header)
                read = ''
            read += line
            header = line[1:].strip()
        else:
            read += line
    if len(read) != 0:
        with open('{}/{}/{}/{}.fna'.format(out_dir, label, sample_id, header), 'w+') as f_read:
            f_read.write(read)
            meta_data_read_list.append(header)
    f_sample.close()
    return meta_data_read_list

def train_test_split(meta_data, train_size_per_class):
    '''
    train test split
    '''
    label_list = sorted(meta_data['label'].unique())
    sample_by_class = {label: meta_data[meta_data['label']==label]['sample_id'].tolist() for label in label_list}
    
    train = []
    test = []
    
    for cls in sample_by_class:
        tmp_list = random.sample(sample_by_class[cls], train_size_per_class)
        train.extend(tmp_list)
        test_list = [sid for sid in sample_by_class[cls] if sid not in train]
        test.extend(test_list)
    
    partition = {'train': train, 'test': test}
    
    return partition

def preprocess_data(opt):
    '''
    preprocessing the data
    '''
    
    if not os.path.exists(opt.out_dir):
        os.makedirs(opt.out_dir)
    
    meta_data = pd.read_csv('{}/meta_data.csv'.format(opt.in_dir), dtype='str')
    
    label_list = sorted(meta_data['label'].unique())
    
    label_dict = {}
    for idx, label in enumerate(label_list):
        label_dict[label] = idx
        
    pickle.dump(label_dict, open('{}/label_dict.pkl'.format(opt.out_dir), 'wb'))
    
    for label in label_list:
        label_dir = '{}/{}'.format(opt.out_dir, label)
        if os.path.exists(label_dir):
            continue
        os.makedirs(label_dir)
        
    read_meta_data = {}
    STEP = meta_data.shape[0] // 10 if meta_data.shape[0] > 10 else 1
    for idx in range(meta_data.shape[0]):
        if idx % STEP == 0:
            logging.info('Processing raw data: {:.1f}% completed.'.format(10 * idx / STEP))
        sample_id, label = meta_data.iloc[idx]['sample_id'], meta_data.iloc[idx]['label']

        read_meta_data[sample_id] = split_sample(sample_id, label, opt.in_dir, opt.out_dir)
    
    min_num_sample = min([meta_data[meta_data['label']==label].shape[0] for label in label_list])
    
    num_train_samples_per_cls = opt.num_train_samples_per_cls
    num_train_samples_per_cls = num_train_samples_per_cls if num_train_samples_per_cls < min_num_sample else min_num_sample
    
    partition = train_test_split(meta_data, num_train_samples_per_cls)
    
    
    sample_to_label = {}
    for idx in range(meta_data.shape[0]):
        sample_id, label = meta_data.iloc[idx]['sample_id'], meta_data.iloc[idx]['label']
        sample_to_label[sample_id] = label

    pickle.dump([sample_to_label, read_meta_data], open('{}/meta_data.pkl'.format(opt.out_dir), 'wb'))
    
    train_list = []
    for sample_id in partition['train']:
        train_list.extend(['{}/{}/{}/{}.fna'.format(opt.out_dir, sample_to_label[sample_id], sample_id, read_id) for read_id in read_meta_data[sample_id]])
    test_list = []
    for sample_id in partition['test']:
        test_list.extend(['{}/{}/{}/{}.fna'.format(opt.out_dir, sample_to_label[sample_id], sample_id, read_id) for read_id in read_meta_data[sample_id]])
    
    read_partition = {'train': train_list, 'test': test_list}
    pickle.dump(read_partition, open('{}/train_test_split.pkl'.format(opt.out_dir), 'wb'))    
    
def preprocess_data_pickle(opt):
    '''
    preprocessing the data
    '''
    
    if not os.path.exists(opt.out_dir):
        os.makedirs(opt.out_dir)
    
    meta_data = pd.read_csv('{}/meta_data.csv'.format(opt.in_dir), dtype='str')
    
    label_list = sorted(meta_data['label'].unique())
    
    label_dict = {}
    for idx, label in enumerate(label_list):
        label_dict[label] = idx
        
    pickle.dump(label_dict, open('{}/label_dict.pkl'.format(opt.out_dir), 'wb'))
    
    for label in label_list:
        label_dir = '{}/{}'.format(opt.out_dir, label)
        if os.path.exists(label_dir):
            continue
        os.makedirs(label_dir)
        
    read_meta_data = {}
    STEP = meta_data.shape[0] // 10 if meta_data.shape[0] > 10 else 1
    for idx in range(meta_data.shape[0]):
        if idx % STEP == 0:
            logging.info('Processing raw data: {:.1f}% completed.'.format(10 * idx / STEP))
        sample_id, label = meta_data.iloc[idx]['sample_id'], meta_data.iloc[idx]['label']

        read_meta_data[sample_id] = fna_to_dict(sample_id, label, opt.in_dir, opt.out_dir)
    
    min_num_sample = min([meta_data[meta_data['label']==label].shape[0] for label in label_list])
    
    num_train_samples_per_cls = opt.num_train_samples_per_cls
    num_train_samples_per_cls = num_train_samples_per_cls if num_train_samples_per_cls < min_num_sample else min_num_sample
    
    partition = train_test_split(meta_data, num_train_samples_per_cls)
    
    
    sample_to_label = {}
    for idx in range(meta_data.shape[0]):
        sample_id, label = meta_data.iloc[idx]['sample_id'], meta_data.iloc[idx]['label']
        sample_to_label[sample_id] = label

    pickle.dump([sample_to_label, read_meta_data], open('{}/meta_data.pkl'.format(opt.out_dir), 'wb'))
    
    train_list = []
    for sample_id in partition['train']:
        train_list.extend([('{}/{}/{}.pkl'.format(opt.out_dir, sample_to_label[sample_id], sample_id), read_id) for read_id in read_meta_data[sample_id]])
    test_list = []
    for sample_id in partition['test']:
        test_list.extend([('{}/{}/{}.pkl'.format(opt.out_dir, sample_to_label[sample_id], sample_id), read_id) for read_id in read_meta_data[sample_id]])
    
    read_partition = {'train': train_list, 'test': test_list}
    pickle.dump(read_partition, open('{}/train_test_split.pkl'.format(opt.out_dir), 'wb'))   