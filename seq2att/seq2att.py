import argparse
import logging
import keras
import pickle
import numpy as np

from .SeqAttModel import SeqAttModel
from .SeqVisualUnit import SeqVisualUnit
from .utils import preprocess_data, preprocess_data_pickle
from .DataGenerator import DataGenerator, DataGeneratorUnlabeled, DataGeneratorPickle, DataGeneratorUnlabeledPickle

from config import Config

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)

def main():
    
    parser = argparse.ArgumentParser(description='seq2att is a command line interface to train a Read2Pheno model on customized 16S rRNA dataset.', prog='seq2att')
    subparsers = parser.add_subparsers(title='subcommands',
                                       description='the following subcommands \
                                    are available: build, train, visualize, default', 
                                    dest='subparser_name')
    # build
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("-m", metavar="metadata",
                        help="metadata used to build training and testing datasets",
                        required=True, type=str)
    # train
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("-m", metavar="metadata",
                        help="metadata used to build training and testing datasets",
                        required=True, type=str)
    # visualize
    visualize_parser = subparsers.add_parser("visualize")
    visualize_parser.add_argument("-m", metavar="metadata",
                        help="metadata used to build training and testing datasets",
                        required=True, type=str)
    # default
    visualize_parser = subparsers.add_parser("default")
    visualize_parser.add_argument("-m", metavar="metadata",
                        help="metadata used to build training and testing datasets",
                        required=True, type=str)
    args = parser.parse_args()
    if args.subparser_name == "build":
        metadata = args.m
        ## TO BE DEVELOPED
        opt = Config()
        ## TO BE DEVELOPED
        preprocess_data_pickle(opt)
    elif args.subparser_name == "train":
        metadata = args.m
        ## TO BE DEVELOPED
        opt = Config()
        ## TO BE DEVELOPED
        label_dict = pickle.load(open('{}/label_dict.pkl'.format(opt.out_dir), 'rb')) 
        sample_to_label, read_meta_data = pickle.load(open('{}/meta_data.pkl'.format(opt.out_dir), 'rb'))
        partition = pickle.load(open('{}/train_test_split.pkl'.format(opt.out_dir), 'rb')) 
        seq_att_model = SeqAttModel(opt)
        training_generator = DataGeneratorPickle(partition['train'], sample_to_label, label_dict, 
                                   dim=(opt.SEQLEN,opt.BASENUM), batch_size=opt.batch_size, shuffle=opt.shuffle)
        testing_generator = DataGeneratorPickle(partition['test'], sample_to_label, label_dict, 
                                   dim=(opt.SEQLEN,opt.BASENUM), batch_size=opt.batch_size, shuffle=opt.shuffle)
        seq_att_model.train_generator(training_generator, n_workers=opt.n_workers)
        seq_att_model.evaluate_generator(testing_generator, n_workers=opt.n_workers)
        
    elif args.subparser_name == "visualize":
        metadata = args.m
        ## TO BE DEVELOPED
        opt = Config()
        ## TO BE DEVELOPED
        prediction, attention_weights, sequence_embedding = seq_att_model.extract_weigths(X_visual)
        idx_to_label = {label_dict[label]: label for label in label_dict}
        seq_visual_unit = SeqVisualUnit(X_visual, y_visual, idx_to_label, taxa_label_list, 
                                        prediction, attention_weights, sequence_embedding, 'Figures')
        seq_visual_unit.plot_embedding()
        seq_visual_unit.plot_attention('Prevotella')
        
    elif args.subparser_name == "default":
        metadata = args.m
        ## TO BE DEVELOPED
        opt = Config()
        ## TO BE DEVELOPED
        preprocess_data_pickle(opt)
        label_dict = pickle.load(open('{}/label_dict.pkl'.format(opt.out_dir), 'rb')) 
        sample_to_label, read_meta_data = pickle.load(open('{}/meta_data.pkl'.format(opt.out_dir), 'rb'))
        partition = pickle.load(open('{}/train_test_split.pkl'.format(opt.out_dir), 'rb')) 
        seq_att_model = SeqAttModel(opt)
        training_generator = DataGeneratorPickle(partition['train'], sample_to_label, label_dict, 
                                   dim=(opt.SEQLEN,opt.BASENUM), batch_size=opt.batch_size, shuffle=opt.shuffle)
        testing_generator = DataGeneratorPickle(partition['test'], sample_to_label, label_dict, 
                                   dim=(opt.SEQLEN,opt.BASENUM), batch_size=opt.batch_size, shuffle=opt.shuffle)
        seq_att_model.train_generator(training_generator, n_workers=opt.n_workers)
        seq_att_model.evaluate_generator(testing_generator, n_workers=opt.n_workers)
        
        metadata = args.m
        ## TO BE DEVELOPED
        opt = Config()
        ## TO BE DEVELOPED
        prediction, attention_weights, sequence_embedding = seq_att_model.extract_weigths(X_visual)
        idx_to_label = {label_dict[label]: label for label in label_dict}
        seq_visual_unit = SeqVisualUnit(X_visual, y_visual, idx_to_label, taxa_label_list, 
                                        prediction, attention_weights, sequence_embedding, 'Figures')
        seq_visual_unit.plot_embedding()
        seq_visual_unit.plot_attention('Prevotella')
        
    
    
if __name__ == "__main__":
    main()