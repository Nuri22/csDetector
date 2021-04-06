try:
    import torch
except (ModuleNotFoundError, ImportError) as e:
    raise ModuleNotFoundError("torch is not currently installed. Run 'pip install convokit[craft]' if you would like to use the CRAFT model.")

import pandas as pd
from convokit.forecaster.CRAFT.CRAFTUtil import loadPrecomputedVoc, batchIterator, CONSTANTS
from .CRAFT.CRAFTNN import initialize_model, makeContextEncoderInput, Predictor
from .forecasterModel import ForecasterModel
import numpy as np
import torch.nn.functional as F
from torch import optim
from sklearn.model_selection import train_test_split
from typing import Dict
import os

default_options = {
    'hidden_size': 500,
    'encoder_n_layers': 2,
    'context_encoder_n_layers': 2,
    'decoder_n_layers': 2,
    'dropout': 0.1,
    'batch_size': 64,
    'clip': 50.0,
    'learning_rate': 1e-5,
    'print_every': 10,
    'train_epochs': 30,
    'validation_size': 0.2,
    'max_length': 80,
    'trained_model_output_filepath': "finetuned_model.tar"
}

# To understand the separation of concerns for the CRAFT files:
# CRAFT/craftNN.py contains the class implementations needed to initialize the CRAFT Neural Network model
# CRAFT/craftUtil.py contains utility methods for manipulating the data for it to be passed to the CRAFT model

class CRAFTModel(ForecasterModel):
    """
    CRAFTModel is one of the Forecaster models that can be used with the Forecaster Transformer.

    By default, CRAFTModel will be initialized with default options

    - hidden_size: 500
    - encoder_n_layers: 2
    - context_encoder_n_layers: 2
    - decoder_n_layers: 2
    - dropout: 0.1
    - batch_size (batch size for computation, i.e. how many (context, reply, id) tuples to use per batch of evaluation): 64
    - clip: 50.0
    - learning_rate: 1e-5
    - print_every: 10
    - train_epochs (number of epochs for training): 30
    - validation_size (percentage of training input data to use as validation): 0.2
    - max_length (maximum utterance length in the dataset): 80

    :param device_type: 'cpu' or 'cuda', default: 'cpu'
    :param model_path: filepath to CRAFT model if loading a custom CRAFT model
    :param options: configuration options for the neural network: uses default options otherwise.
    :param forecast_attribute_name: name of DataFrame column containing predictions, default: "prediction"
    :param forecast_prob_attribute_name: name of DataFrame column containing prediction scores, default: "score"
    """

    def __init__(self, device_type: str = 'cpu',
                 model_path: str = None,
                 options: Dict = None,
                 forecast_attribute_name: str = "prediction",
                 forecast_feat_name=None,
                 forecast_prob_attribute_name: str = "pred_score",
                 forecast_prob_feat_name=None):

        super().__init__(forecast_attribute_name=forecast_attribute_name,
                         forecast_feat_name=forecast_feat_name,
                         forecast_prob_attribute_name=forecast_prob_attribute_name,
                         forecast_prob_feat_name=forecast_prob_feat_name)
        assert device_type in ['cuda', 'cpu']
        # device: controls GPU usage: 'cuda' to enable GPU, 'cpu' to run on CPU only.
        self.device = torch.device(device_type)
        self.device_type = device_type
        # voc: the vocabulary object (convokit.forecaster.craftUtil.Voc) used by predictor.
        # Used to convert text data into numerical input for CRAFT.
        self.voc = loadPrecomputedVoc("wikiconv", CONSTANTS['WORD2INDEX_URL'], CONSTANTS['INDEX2WORD_URL'])

        if options is None:
            self.options = default_options
        else:
            for k, v in default_options.items():
                if k not in options:
                    options[k] = v
            self.options = options
        print("Initializing CRAFT model with options:")
        print(self.options)

        if model_path is not None:
            if not os.path.isfile(model_path) or not model_path.endswith(".tar"):
                print("Could not find CRAFT model tar file at: {}".format(model_path))
                model_path = None
        self.predictor: Predictor = initialize_model(model_path, self.voc, self.device, self.device_type,
                                                     self.options['hidden_size'],
                                                     self.options['encoder_n_layers'],
                                                     self.options['dropout'],
                                                     self.options['context_encoder_n_layers'])


    def _evaluate_batch(self, predictor, input_batch, dialog_lengths,
                        dialog_lengths_list, utt_lengths, batch_indices, dialog_indices, true_batch_size):
        """
        Helper for _evaluate_dataset. Runs CRAFT evaluation on a single batch; _evaluate_dataset calls this helper iteratively to get results for the entire dataset.

        :param predictor: the trained CRAFT model to use, provided as a PyTorch Model instance.
        :param input_batch: the batch to run CRAFT on (produced by convokit.forecaster.craftUtil.batchIterator, formatted as a batch of utterances)
        :param dialog_lengths: how many comments are in each conversation in this batch, as a PyTorch Tensor
        :param dialog_lengths_list: same as dialog_lengths, but as a Python List
        :param utt_lengths: for each conversation, records the number of tokens in each utterance of the conversation
        :param batch_indices: used by CRAFT to reconstruct the original dialog batch from the given utterance batch. Records which dialog each utterance originally came from.
        :param dialog_indices: used by CRAFT to reconstruct the original dialog batch from the given utterance batch. Records where in the dialog the utterance originally came from.
        :param true_batch_size: number of dialogs in the original dialog batch this utterance batch was generated from.

        :return: per-utterance scores and binarized predictions.
        """
        # Set device options
        input_batch = input_batch.to(self.device)
        dialog_lengths = dialog_lengths.to(self.device)
        utt_lengths = utt_lengths.to(self.device)
        # Predict future attack using predictor
        scores = predictor(input_batch, dialog_lengths, dialog_lengths_list, utt_lengths,
                           batch_indices, dialog_indices, true_batch_size, self.options['max_length'])
        predictions = (scores > 0.5).float()
        return predictions, scores

    def _evaluate_dataset(self, predictor, dataset):
        """
        Run a trained CRAFT model over an entire dataset in a batched fashion.

        :param predictor: the trained CRAFT model to use, provided as a PyTorch Model instance.
        :param dataset: the dataset to evaluate on, formatted as a list of (context, reply, id_of_reply) tuples.
        :return: a DataFrame, indexed by utterance ID, of CRAFT scores for each utterance, and the corresponding binary prediction.
        """
        # create a batch iterator for the given data
        batch_iterator = batchIterator(self.voc, dataset, self.options['batch_size'], shuffle=False)
        # find out how many iterations we will need to cover the whole dataset
        n_iters = len(dataset) // self.options['batch_size'] + int(len(dataset) % self.options['batch_size'] > 0)
        output_df = {
            "id": [],
            self.forecast_attribute_name: [],
            self.forecast_prob_attribute_name: []
        }
        for iteration in range(1, n_iters+1):
            batch, batch_dialogs, true_batch_size = next(batch_iterator)
            # Extract fields from batch
            input_variable, dialog_lengths, utt_lengths, batch_indices, dialog_indices, \
                labels, batch_ids, target_variable, mask, max_target_len = batch
            dialog_lengths_list = [len(x) for x in batch_dialogs]
            # run the model
            predictions, scores = self._evaluate_batch(predictor, input_variable, dialog_lengths, dialog_lengths_list, utt_lengths,
                                                       batch_indices, dialog_indices, true_batch_size)

            # format the output as a dataframe (which we can later re-join with the corpus)
            for i in range(true_batch_size):
                utt_id = batch_ids[i]
                pred = predictions[i].item()
                score = scores[i].item()
                output_df["id"].append(utt_id)
                output_df[self.forecast_attribute_name].append(pred)
                output_df[self.forecast_prob_attribute_name].append(score)

            print("Iteration: {}; Percent complete: {:.1f}%".format(iteration, iteration / n_iters * 100))

        return pd.DataFrame(output_df).set_index("id")

    def _train_NN(self, input_variable, dialog_lengths, dialog_lengths_list, utt_lengths, batch_indices, dialog_indices, labels, # input/output arguments
              encoder, context_encoder, attack_clf,                                                                    # network arguments
              encoder_optimizer, context_encoder_optimizer, attack_clf_optimizer,                                      # optimization arguments
              batch_size, clip):                                                                # misc arguments

        # Zero gradients
        encoder_optimizer.zero_grad()
        context_encoder_optimizer.zero_grad()
        attack_clf_optimizer.zero_grad()

        # Set device options
        input_variable = input_variable.to(self.device)
        dialog_lengths = dialog_lengths.to(self.device)
        utt_lengths = utt_lengths.to(self.device)
        labels = labels.to(self.device)

        # Forward pass through utterance encoder
        _, utt_encoder_hidden = encoder(input_variable, utt_lengths)

        # Convert utterance encoder final states to batched dialogs for use by context encoder
        context_encoder_input = makeContextEncoderInput(utt_encoder_hidden, dialog_lengths_list, batch_size, batch_indices, dialog_indices)

        # Forward pass through context encoder
        context_encoder_outputs, _ = context_encoder(context_encoder_input, dialog_lengths)

        # Forward pass through classifier to get prediction logits
        logits = attack_clf(context_encoder_outputs, dialog_lengths)

        # Calculate loss
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        # Perform backpropatation
        loss.backward()

        # Clip gradients: gradients are modified in place
        _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
        _ = torch.nn.utils.clip_grad_norm_(context_encoder.parameters(), clip)
        _ = torch.nn.utils.clip_grad_norm_(attack_clf.parameters(), clip)

        # Adjust model weights
        encoder_optimizer.step()
        context_encoder_optimizer.step()
        attack_clf_optimizer.step()

        return loss.item()

    def _validate(self, predictor, dataset):
        # create a batch iterator for the given data
        batch_iterator = batchIterator(self.voc, dataset, self.options['batch_size'], shuffle=False)
        # find out how many iterations we will need to cover the whole dataset
        n_iters = len(dataset) // self.options['batch_size'] + int(len(dataset) % self.options['batch_size'] > 0)
        # containers for full prediction results so we can compute accuracy at the end
        all_preds = []
        all_labels = []
        for iteration in range(1, n_iters+1):
            batch, batch_dialogs, true_batch_size = next(batch_iterator)
            # Extract fields from batch
            input_variable, dialog_lengths, utt_lengths, batch_indices, dialog_indices, \
                batch_labels, batch_ids, target_variable, mask, max_target_len = batch
            dialog_lengths_list = [len(x) for x in batch_dialogs]
            # run the model
            predictions, scores = self._evaluate_batch(predictor, input_variable, dialog_lengths, dialog_lengths_list,
                                                       utt_lengths, batch_indices, dialog_indices,
                                                       true_batch_size)
            # aggregate results for computing accuracy at the end
            all_preds += [p.item() for p in predictions]
            all_labels += [l.item() for l in batch_labels]
            print("Iteration: {}; Percent complete: {:.1f}%".format(iteration, iteration / n_iters * 100))

        # compute and return the accuracy
        return (np.asarray(all_preds) == np.asarray(all_labels)).mean()

    def _train_iters(self, train_pairs, val_pairs, encoder, context_encoder, attack_clf,
                     encoder_optimizer, context_encoder_optimizer, attack_clf_optimizer, embedding,
                     n_iteration, validate_every):

        # create a batch iterator for training data
        batch_iterator = batchIterator(self.voc, train_pairs, self.options['batch_size'])

        # Initializations
        print('Initializing ...')
        start_iteration = 1
        print_loss = 0

        # Training loop
        print("Training...")
        # keep track of best validation accuracy - only save when we have a model that beats the current best
        best_acc = 0
        for iteration in range(start_iteration, n_iteration + 1):
            training_batch, training_dialogs, true_batch_size = next(batch_iterator)
            # Extract fields from batch
            input_variable, dialog_lengths, utt_lengths, batch_indices, dialog_indices, \
            labels, batch_ids, target_variable, mask, max_target_len = training_batch
            dialog_lengths_list = [len(x) for x in training_dialogs]

            # Run a training iteration with batch
            loss = self._train_NN(input_variable, dialog_lengths, dialog_lengths_list, utt_lengths, batch_indices, dialog_indices, labels, # input/output arguments
                         encoder, context_encoder, attack_clf,                                                                    # network arguments
                         encoder_optimizer, context_encoder_optimizer, attack_clf_optimizer,                                      # optimization arguments
                         true_batch_size, self.options['clip'])                                                                                   # misc arguments
            print_loss += loss

            # Print progress
            if iteration % self.options['print_every'] == 0:
                print_loss_avg = print_loss / self.options['print_every']
                print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration, iteration / n_iteration * 100, print_loss_avg))
                print_loss = 0

            # Evaluate on validation set
            if iteration % validate_every == 0:
                print("Validating!")
                # put the network components into evaluation mode
                encoder.eval()
                context_encoder.eval()
                attack_clf.eval()

                predictor = Predictor(encoder, context_encoder, attack_clf)
                accuracy = self._validate(predictor, val_pairs)
                print("Validation set accuracy: {:.2f}%".format(accuracy * 100))

                # keep track of our best model so far
                if accuracy > best_acc:
                    print("Validation accuracy better than current best; saving model...")
                    best_acc = accuracy
                    torch.save({
                        'iteration': iteration,
                        'en': encoder.state_dict(),
                        'ctx': context_encoder.state_dict(),
                        'atk_clf': attack_clf.state_dict(),
                        'en_opt': encoder_optimizer.state_dict(),
                        'ctx_opt': context_encoder_optimizer.state_dict(),
                        'atk_clf_opt': attack_clf_optimizer.state_dict(),
                        'loss': loss,
                        'voc_dict': self.voc.__dict__,
                        'embedding': embedding.state_dict()
                    }, self.options['trained_model_output_filepath'])

                # put the network components back into training mode
                encoder.train()
                context_encoder.train()
                attack_clf.train()

    def train(self, id_to_context_reply_label):
        ids = list(id_to_context_reply_label)
        train_pair_ids, val_pair_ids = train_test_split(ids, test_size=self.options['validation_size'])
        train_pairs = [id_to_context_reply_label[pair_id] for pair_id in train_pair_ids]
        val_pairs = [id_to_context_reply_label[pair_id] for pair_id in val_pair_ids]
        # Compute the number of training iterations we will need in order to achieve the number of epochs specified in the settings at the start of the notebook
        n_iter_per_epoch = len(train_pairs) // self.options['batch_size'] + int(len(train_pairs) % self.options['batch_size'] == 1)
        n_iteration = n_iter_per_epoch * self.options['train_epochs']

        # Put dropout layers in train mode
        self.predictor.encoder.train()
        self.predictor.context_encoder.train()
        self.predictor.classifier.train()

        # Initialize optimizers
        print('Building optimizers...')
        encoder_optimizer = optim.Adam(self.predictor.encoder.parameters(), lr=self.options['learning_rate'])
        context_encoder_optimizer = optim.Adam(self.predictor.context_encoder.parameters(),
                                               lr=self.options['learning_rate'])
        attack_clf_optimizer = optim.Adam(self.predictor.classifier.parameters(), lr=self.options['learning_rate'])

        # Run training iterations, validating after every epoch
        print("Starting Training!")
        print("Will train for {} iterations".format(n_iteration))
        self._train_iters(train_pairs, val_pairs, self.predictor.encoder, self.predictor.context_encoder,
                          self.predictor.classifier, encoder_optimizer, context_encoder_optimizer, attack_clf_optimizer,
                          self.predictor.encoder.embedding, n_iteration, n_iter_per_epoch)


    def forecast(self, id_to_context_reply_label):
        """
        Compute forecasts and forecast scores for the given dictionary of utterance id to (context, reply) pairs. Return the values in a DataFrame.

        :param id_to_context_reply_label: dict mapping utterance id to (context, reply, label)
        :return: a pandas DataFrame
        """
        dataset = [(context, reply, label, id_) for id_, (context, reply, label) in id_to_context_reply_label.items()]
        return self._evaluate_dataset(self.predictor, dataset)
