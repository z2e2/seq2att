import logging
import matplotlib
import matplotlib.pyplot as plt
from .sequence_attention_model import sequence_attention_model
from keras import backend as K

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)

font = {'family': 'sans-serif', # Helvetica
        'size'   : 20}
matplotlib.rc('font', **font)

class SeqAttModel:
    '''
    Attention based sequence model
    '''
    def __init__(self, opt):
        '''
        Model initialization
        '''
        self.opt = opt
        self.model = sequence_attention_model(self.opt)
        self.history = None
        logging.info('Model initialized.')
        
    def train(self, X, y):
        '''
        Model training
        '''
        logging.info('Training started: train the model on {} sequences'.format(X.shape[0]))
        self.history = self.model.fit(X, y, batch_size=self.opt.batch_size, epochs=self.opt.epochs, verbose=self.opt.verbose)
        train_loss, train_acc = self.model.evaluate(X, y, batch_size=self.opt.batch_size, verbose=self.opt.verbose)
        logging.info('Training completed: training accuracy is {:.4f}.'.format(train_acc))
    
    def train_generator(self, training_generator, n_workers):
        '''
        Model training with a generator
        '''
        logging.info('Training started:')
        self.history = self.model.fit_generator(generator=training_generator,
                                                use_multiprocessing=True,
                                                workers=n_workers, verbose=True)
        eval_loss, eval_acc = self.model.evaluate_generator(generator=training_generator,
                                                            use_multiprocessing=True,
                                                            workers=n_workers)
        logging.info('Evaluation completed: evaluation accuracy is {:.4f}.'.format(eval_acc))
        
    def predict(self, X):
        '''
        Model predicting
        '''
        logging.info('Predicting started: predict {} sequences'.format(X.shape[0]))
        pred = self.model.predict(X, batch_size=self.opt.batch_size, verbose=self.opt.verbose)
        logging.info('Predicting completed.')
        return pred
    
    def predict_generator(self, pred_generator, n_workers):
        '''
        Model evaluation with a generator
        '''
        logging.info('Predicting started:')
        pred = self.model.predict_generator(generator=pred_generator,
                                            use_multiprocessing=True,
                                            workers=n_workers, verbose=self.opt.verbose)
        logging.info('Predicting completed.')
        return pred
        
    def evaluate(self, X, y):
        '''
        Model evaluation
        '''
        logging.info('Evaluation started: evalute {} sequences'.format(X.shape[0]))
        eval_loss, eval_acc = self.model.evaluate(X, y, batch_size=self.opt.batch_size, verbose=self.opt.verbose)
        logging.info('Evaluation completed: evaluation accuracy is {:.4f}.'.format(eval_acc))
        return eval_acc
    
    def evaluate_generator(self, test_generator, n_workers):
        '''
        Model evaluation with a generator
        '''
        logging.info('Evaluation started:')
        eval_loss, eval_acc = self.model.evaluate_generator(generator=test_generator,
                                                            use_multiprocessing=True,
                                                            workers=n_workers)
        logging.info('Evaluation completed: evaluation accuracy is {:.4f}.'.format(eval_acc))
        return eval_acc
    
    def save(self, name):
        '''
        Model save
        '''
        self.model.save_weights(name)
        
    def load(self, name):
        '''
        Model load
        '''
        self.model.load_weights(name)
        
    def _extract_intermediate_output(self):
        '''
        helper function to extract intermediate layer output
        '''
        inputs = []
        inputs.extend(self.model.inputs)
        outputs = []
        outputs.extend(self.model.outputs)

        attention = self.model.get_layer('att_weights')
        embedding = self.model.get_layer('att_emb')

        outputs.append(attention.output)
        outputs.append(embedding.output)

        extract_intermediate_layer = K.function(inputs, outputs)
        return extract_intermediate_layer
    
    def extract_weigths(self, X):
        '''
        extract sequence attention weigths and sequence embedding from the model for input sequences, X.
        '''
        logging.info('Weights extraction started: extract weights for {} sequences'.format(X.shape[0]))
        extract_intermediate_layer = self._extract_intermediate_output()
        prediction, attention_weights, sequence_embedding = extract_intermediate_layer([X])
        logging.info('Weights extraction completed.')
        return prediction, attention_weights, sequence_embedding
    
    def visualize_training_history(self):
        '''
        Visualize training history
        '''
        if self.history is None:
            logging.info('Model training history does not exist.')
            return
        plt.figure(figsize=(8, 6))
        if 'val_loss' in self.history.history:
            plt.plot(self.history.history['loss'], 'o--', color='k', label='val_loss')
            plt.title('Validation dataset loss')
        else:
            plt.plot(self.history.history['loss'], 'o--', color='k', label='train_loss')
            plt.title('Training dataset loss')

        plt.xlabel('Epochs')
        plt.ylabel('loss')
        plt.show()




