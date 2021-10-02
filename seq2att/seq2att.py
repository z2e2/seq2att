import argparse
import logging
import keras
import pickle
import numpy as np
import yaml
from .SeqAttModel import SeqAttModel
from .SeqVisualUnit import SeqVisualUnit
from .utils import preprocess_data, preprocess_data_pickle
from .DataGenerator import DataGenerator, DataGeneratorUnlabeled, DataGeneratorPickle, DataGeneratorUnlabeledPickle
from .config import Config

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)

def main():
    
    parser = argparse.ArgumentParser(description='seq2att is a command line interface to train a Read2Pheno model on customized 16S rRNA dataset.', prog='seq2att')
    subparsers = parser.add_subparsers(title='subcommands',
                                       description='the following subcommands \
                                    are available: build, train, attention, default', 
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
    visualize_parser = subparsers.add_parser("attention")
    visualize_parser.add_argument("-m", metavar="metadata",
                        help="metadata used to build training and testing datasets",
                        required=True, type=str)
    visualize_parser.add_argument("-taxa", metavar="taxonomic labels",
                        help="path to the taxonomic labels of the visualization dataset used to color code different reads by their taxonomic labels",
                        required=True, type=str)
    visualize_parser.add_argument("-data", metavar="visualization dataset",
                        help="path to visualization dataset the user seleced for visualization and model interpretation",
                        required=True, type=str)
    visualize_parser.add_argument("-nanme", metavar="taxon name to be visualized",
                        help="a certain taxon that the user wants to visualize for attention weights interpretation",
                        required=True, type=str)
    # default
    default_parser = subparsers.add_parser("default")
    default_parser.add_argument("-m", metavar="metadata",
                        help="metadata used to build training and testing datasets",
                        required=True, type=str)
    args = parser.parse_args()
    if args.subparser_name == "build":
        metadata = args.m
        ## LOAD METADATA
        opt = Config()
        config_dict = yaml.safe_load(open(metadata))
        for key in config_dict:
            setattr(opt, key, config_dict[key])
        ## LOAD METADATA
        preprocess_data_pickle(opt)
        logging.info('Program completed and preprocessed data were saved to {}'.format(opt.out_dir))
    elif args.subparser_name == "train":
        metadata = args.m
        ## LOAD METADATA
        opt = Config()
        config_dict = yaml.safe_load(open(metadata))
        for key in config_dict:
            setattr(opt, key, config_dict[key])
        ## LOAD METADATA
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
        seq_att_model.save('{}/{}'.format(opt.save_model_path, opt.model_name))
        logging.info('Program completed and model was saved to {}'.format(opt.save_model_path))
    elif args.subparser_name == "attention":
        metadata = args.m
        visualdata = args.data
        taxadata = args.taxa
        taxaname = args.name
        X_visual, y_visual = pickle.load(open(visualdata, 'rb'))
        taxa_label_list = pickle.load(open(taxadata, 'rb'))
        ## LOAD METADATA
        opt = Config()
        config_dict = yaml.safe_load(open(metadata))
        for key in config_dict:
            setattr(opt, key, config_dict[key])
        ## LOAD Model
        seq_att_model = SeqAttModel(opt)
        seq_att_model.load('{}/{}'.format(opt.save_model_path, opt.model_name))
        label_dict = pickle.load(open('{}/label_dict.pkl'.format(opt.out_dir), 'rb'))
        prediction, attention_weights, sequence_embedding = seq_att_model.extract_weigths(X_visual)
        idx_to_label = {label_dict[label]: label for label in label_dict}
        pickle.dump([prediction, attention_weights, sequence_embedding, idx_to_label], open(opt.attention_weights_output, 'wb'))
        logging.info('Program completed and attention weights were saved to {}'.format(opt.attention_weights_output))
#         seq_visual_unit = SeqVisualUnit(X_visual, y_visual, idx_to_label, taxa_label_list, 
#                                         prediction, attention_weights, sequence_embedding, 'Figures')
#         seq_visual_unit.plot_embedding()
#         seq_visual_unit.plot_attention(taxa_name)
        
    elif args.subparser_name == "default":
        metadata = args.m
        ## LOAD METADATA
        opt = Config()
        config_dict = yaml.safe_load(open(metadata))
        for key in config_dict:
            setattr(opt, key, config_dict[key])
        ## LOAD METADATA
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
        seq_att_model.save(opt.save_model_path)
        logging.info('Program completed and model was saved to {}'.format(opt.save_model_path))
    
    
if __name__ == "__main__":
    main()
