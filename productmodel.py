#!/usr/bin/env python

import warnings
import pandas as pd
import numpy as np
import itertools
import os
import pickle
import logging
import re
import sys
import xlsxwriter
from pycm import *
from keras.layers import Input, Dense, Dropout, Embedding, Flatten, Conv1D, SpatialDropout1D, MaxPooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical, plot_model
import keras
from keras_tqdm import TQDMNotebookCallback
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scriptine import run

sys.path.append(os.getcwd())
from product_classification.lib import get_df_from_file # substitute with your data loading func

# Disable future warning of scipy and pandas slicing
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

logger = logging.getLogger('ProductTrainer')
logging.basicConfig(format='%(asctime)s - %(levelname)s:%(message)s', level=logging.INFO)


class ProductModel:
    artefact_path = "product_classification/product_prediction/artefacts/"
    empty_token = 'emptystring'
    all_flds = ['tag_Group', 'tag_Category', 'tag_Subcategory', 'tag_Type', 'tag_Brand', 'tag_Mixer_combination',
                'tag_Container', 'tag_Volume', 'tag_SKU']

    # venues with many parser errors in descriptions
    excluded_venues = [78, 79, 80, 1535]

    def __init__(self, data, from_path=False):
        """
        The ProductModel class allows you to train a model for multiple columns of the products dimension.
        To add a column, add the name of the column to above mentioned 'all_flds' variable, as well as the
        _get_trainable_rows() function, where you have to specify which other columns must have a value.
        Lastly, if the value can be empty, add the field to _preprocess() to be filled with our emptytoken
        :param data: a dataframe or path for the data
        :param from_path: if using a path, this must be set to true.
        """
        if from_path:
            self.data = get_df_from_file(data)
        else:
            self.data = data
        if not isinstance(self.data, pd.DataFrame):
            raise TypeError('data should be pd.DataFrame. If loading data with path, pass from_path=True')

        self.results = []
        self.dataset_tokenizer = None
        self.label_mapping = None
        self.reverse_label_mapping = None
        self.label_tokenizer = None
        self.reverse_label_tokenizer = None
        self.predictions_df = None
        self.evaluate_df = None
        self.eval_data = None

    def fit(self, train_flds=None, batch_size=256, epochs=150, val_perc=0.1, verbose=0, nr_samples=0):
        logger.info(f'Verbose={verbose}, use 1 for output each batch and 2 for output each epoch.')

        if train_flds is None:
            train_flds = []
        data = self.data[:nr_samples] if nr_samples else self.data
        df = self._preprocess_data(data, self.empty_token, mode='train')
        if not os.path.exists(self.artefact_path):
            os.mkdir(self.artefact_path)
        total_df, self.dataset_tokenizer = self._add_tokens(df, 'description', 'data_tokenizer')

        # We use both a label mapping and a label tokenizer because the tokenizer strips characters from strings before
        # creating a token. Because we want to get the exact same strings as the input, we map the strings to a cleaned
        # string that is saved in the label_mapping. For example 'Bitter/Kruidenbitter' is transformed into
        # 'bitter kruidenbitter' by the label mapping which is then turned into a number by the Tokenizer.
        # The label_mapping, reverse_label_mapping and reverse_label_tokenizer are dictionaries, while the
        # label_tokenizer is a Keras Tokenizer object.
        self.label_mapping, self.reverse_label_mapping = self._get_label_mapping(total_df)
        self.label_tokenizer, self.reverse_label_tokenizer = self._get_label_tokenizer(df)
        train_flds = [x for x in self.all_flds if x in train_flds] if train_flds else self.all_flds
        for train_fld in train_flds:
            self._train_fld(total_df, train_fld, batch_size, epochs, val_perc, verbose)

        logger.info(
            "############################################# RESULTS #############################################\n")
        for key, value in self.results:
            logger.info(key + ": " + str(value))
        return self.results, len(total_df)

    def predict(self, pred_flds=None, nr_samples=0):
        if pred_flds is None:
            pred_flds = []
        data = self.data[:nr_samples] if nr_samples else self.data
        data = self._preprocess_data(data, self.empty_token, mode='predict')
        total_df, self.dataset_tokenizer = self._add_tokens(data, 'description', 'data_tokenizer', is_pred=True)
        self.label_mapping, self.reverse_label_mapping = self._get_label_mapping(is_pred=True)
        self.label_tokenizer, self.reverse_label_tokenizer = self._get_label_tokenizer(is_pred=True)
        pred_flds = [x for x in self.all_flds if x in pred_flds] if pred_flds else self.all_flds

        for class_type in pred_flds:
            total_df = self._predict_fld(total_df, class_type)
        self.predictions_df = self._cleanup(total_df)
        return self.predictions_df

    def evaluate(self, eval_flds=None, nr_samples=0, plot=False):
        if eval_flds is None:
            eval_flds = []
        data = self.data[:nr_samples] if nr_samples else self.data
        data = self._preprocess_data(data, self.empty_token, mode='evaluate')
        total_df, self.dataset_tokenizer = self._add_tokens(data, 'description', 'data_tokenizer', is_pred=True)
        self.label_mapping, self.reverse_label_mapping = self._get_label_mapping(is_pred=True)
        self.label_tokenizer, self.reverse_label_tokenizer = self._get_label_tokenizer(is_pred=True)
        eval_flds = [x for x in self.all_flds if x in eval_flds] if eval_flds else self.all_flds
        self.eval_data = {'high_conf': {}, 'low_conf': {}}

        for class_type in eval_flds:
            total_df = self._eval_fld(total_df, class_type, plot=plot)
        self.evaluate_df = self._cleanup(total_df)
        export_location = 'data/difficult_samples.xlsx'
        logger.info(f'Writing confusing samples to: {export_location}')
        writer = pd.ExcelWriter(export_location, engine='xlsxwriter')
        for key1, value in self.eval_data.items():
            for key2, value2 in value.items():
                value2.to_excel(writer, sheet_name=f'{key1} {key2}', index=False)
            writer.save()
        return self.evaluate_df

    def _train_fld(self, total_df, class_type, batch_size, epochs, val_perc, verbose=0):
        """
        The training and evaluation segment for the model
        :param total_df: All data and additional data for classification
        :param class_type: what kind of prediction is being done (e.g. group or category)
        :return: Nothing, all data is by reference
        """
        logger.info(
            f"\n################################ {class_type.upper()} TRAIN START ################################")

        x = self._remove_columns_not_for_classification(total_df, class_type)
        x[class_type] = total_df[class_type]
        x = self._get_trainable_rows(x, class_type)

        # y has to be gotten between the two _remove_columns_not_for_classification, as this removes the target labels.
        y = self._label2categorical(x, class_type)
        x = self._tag_columns_to_tokens(x)
        x = self._remove_columns_not_for_classification(x, class_type)
        assert (len(x) > 0)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, shuffle=True)
        vocab_size = len(self.dataset_tokenizer.word_index) + 1
        num_class = len(y_train[0])
        input_length = len(x_train.iloc[0])
        filepath = f"{self.artefact_path}cnn-weights-{class_type}.h5"
        logger.debug(f'Weights location: {filepath}')

        logger.info("Nr. samples training: {}".format(len(x)))
        logger.info(f"vocab_size: {vocab_size}\n"
                    f"num_class: {num_class}\n"
                    f"input_length: {input_length}\n"
                    f"filepath: {filepath}\n")

        keras.backend.clear_session()
        model = cnn_model(input_length=input_length, vocab_size=vocab_size, num_class=num_class, class_type=class_type)

        if class_type == 'tag_Group':
            model.summary()

        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=20, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, min_delta=0.002,
                                      verbose=1)
        checkpointer = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
        # tqdmcallback = TQDMNotebookCallback(leave_inner=True)
        # tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1,  batch_size=128, embeddings_freq=1,
        #                           embeddings_layer_names=['embedding']
        #
        #                           )

        history = model.fit(x_train,
                            y_train,
                            batch_size=batch_size,
                            verbose=verbose,
                            validation_split=val_perc,
                            shuffle=True,
                            epochs=epochs,
                            callbacks=[checkpointer, early_stopping, reduce_lr,
                                       # tqdmcallback
                                       ])

        results = pd.DataFrame({'epochs': history.epoch, 'accuracy': history.history['acc'],
                                'validation_accuracy': history.history['val_acc']})
        logger.info("\nMean train acc: {}\nMean val acc: {}\nMax train acc: {}\nMax val acc: {}".format(
            results['accuracy'].mean(),
            results['validation_accuracy'].mean(),
            results['accuracy'].max(),
            results['validation_accuracy'].max()
        ))
        predicted = model.predict(x_test)
        predicted_num = np.argmax(predicted, axis=1)
        actual_num = np.argmax(y_test, axis=1)
        test_set_acc = accuracy_score(actual_num, predicted_num)
        logger.info("{} test set accuracy: {}".format(class_type, test_set_acc))
        self.results.append(('{} test set accuracy'.format(class_type), test_set_acc))

        actual_classes = [self.reverse_label_tokenizer.get(x) for x in actual_num]
        pred_classes = [self.reverse_label_tokenizer.get(x) for x in predicted_num]
        if class_type != 'tag_Brand':
            cm = ConfusionMatrix(actual_vector=actual_classes, predict_vector=pred_classes)
            cm.save_html(f'product_classification/evaluation/evaluation_{class_type}')

        logger.info(
            f"\n################################ {class_type.upper()} TRAIN END ################################\n")

    def _predict_fld(self, total_df, class_type):
        logger.info(
            f"\n############################## {class_type.upper()} PREDICTION START #############################")

        x = self._remove_columns_not_for_classification(total_df, class_type)
        x[class_type] = total_df[class_type]
        self._fill_na_cells_with_pred(x, total_df, class_type)
        x = self._tag_columns_to_tokens(x)
        x = self._get_trainable_rows(x, class_type, is_pred=True)
        x = self._remove_columns_not_for_classification(x, class_type)
        if len(x) > 0:
            vocab_size = len(self.dataset_tokenizer.word_index) + 1
            num_class = len(self.label_tokenizer.word_index) + 1
            input_length = len(x.iloc[0])
            filepath = f"{self.artefact_path}cnn-weights-{class_type}.h5"
            logger.debug(f'Weights location: {filepath}')
            logger.info("Nr. samples predicting: {}".format(len(x)))
            logger.info(f"vocab_size: {vocab_size}\n"
                        f"num_class: {num_class}\n"
                        f"input_length: {input_length}\n"
                        f"filepath: {filepath}\n")

            assert (os.path.isfile(filepath))
            keras.backend.clear_session()
            model = cnn_model(input_length=input_length, vocab_size=vocab_size, num_class=num_class,
                              class_type=class_type)
            model.load_weights(filepath)
            predicted = model.predict(x)

            predicted_conf = [max(x) for x in predicted]
            predicted_num = np.argmax(predicted, axis=1)
            x[f'{class_type}_pred'] = predicted_num
            x[f'{class_type}_pred_conf'] = predicted_conf
            self._num2labels(x, f'{class_type}_pred')
            total_df[f'{class_type}_pred'] = x[f'{class_type}_pred']
            total_df[f'{class_type}_pred_conf'] = x[f'{class_type}_pred_conf']
        else:
            total_df[f'{class_type}_pred'] = None
            total_df[f'{class_type}_pred_conf'] = None
            logger.info("{} samples to classify was 0".format(class_type.capitalize()))
        logger.info(
            f"\n############################## {class_type.upper()} PREDICTION END #############################\n")
        return total_df

    def _eval_fld(self, total_df, class_type, plot=False):
        logger.info(
            f"\n############################### {class_type.upper()} EVALUATE START ##############################")
        x = self._remove_columns_not_for_classification(total_df, class_type)
        x[class_type] = total_df[class_type]
        self._fill_na_cells_with_pred(x, total_df, class_type)
        x = self._get_trainable_rows(x, class_type)

        # This step must be before the _remove_columns_not_for_classification(), because this removes the target labels.
        # This step also returns and x, because some rows can have (incorrect) labels that are not in the
        # label_tokenizer.
        y, x = self._label2categorical(x, class_type, is_eval=True)
        x = self._tag_columns_to_tokens(x)
        x = self._remove_columns_not_for_classification(x, class_type)

        assert (len(x) > 0)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0, shuffle=True)

        vocab_size = len(self.dataset_tokenizer.word_index) + 1
        num_class = len(y_train[0])
        input_length = len(x_train.iloc[0])
        filepath = f"{self.artefact_path}cnn-weights-{class_type}.h5"
        logger.debug(f"Weights location: {filepath}")

        logger.info(f"Nr. evaluation samples: {len(x)}")
        logger.info(f"vocab_size: {vocab_size}\n"
                    f"num_class: {num_class}\n"
                    f"input_length: {input_length}\n"
                    f"filepath: {filepath}\n")
        assert (os.path.isfile(filepath))
        keras.backend.clear_session()
        eval_model = cnn_model(input_length=input_length, vocab_size=vocab_size, num_class=num_class,
                               class_type=class_type)
        if plot:
            plot_model(eval_model,
                       to_file=f'product_classification/product_prediction/model_visualizations/model_{class_type}.png')
        eval_model.load_weights(filepath)
        predicted = eval_model.predict(x_train)
        predicted_num = np.argmax(predicted, axis=1)
        actual_num = np.argmax(y_train, axis=1)
        total_acc = accuracy_score(actual_num, predicted_num)
        logger.info(f"{class_type} accuracy: {total_acc}")
        self.results.append((f'{class_type} accuracy', total_acc))

        predicted_conf = [max(x) for x in predicted]
        x[f'{class_type}_pred'] = predicted_num
        x[f'{class_type}_pred_conf'] = predicted_conf
        self._num2labels(x, f'{class_type}_pred')
        total_df[f'{class_type}_pred'] = x[f'{class_type}_pred']
        total_df[f'{class_type}_pred_conf'] = x[f'{class_type}_pred_conf']

        actual_classes = [self.reverse_label_tokenizer.get(x) for x in actual_num]
        pred_classes = [self.reverse_label_tokenizer.get(x) for x in predicted_num]
        if class_type != 'tag_Brand':
            cm = ConfusionMatrix(actual_vector=actual_classes, predict_vector=pred_classes)
            cm.save_html(f'product_classification/evaluation/evaluation_{class_type}')
        eval_df = pd.DataFrame({'actual_classes': actual_classes, 'pred_classes': pred_classes,
                                'confidence': np.max(predicted, axis=1)}, index=x_train.index)
        self.eval_data['high_conf'][class_type] = self._conf_eval(total_df, eval_df, correct=False, top_n=1000,
                                                                  highest=True)
        self.eval_data['low_conf'][class_type] = self._conf_eval(total_df, eval_df, correct=True, top_n=1000,
                                                                 highest=False)

        logger.info(
            f"\n############################### {class_type.upper()} EVALUATE END ##############################\n")
        return total_df

    def _cleanup(self, df):
        regex = re.compile(r"x\d")
        feature_cols = list(filter(regex.match, df.columns))
        df = df.drop(feature_cols, axis=1)
        df = self._fix_column_order_and_dtypes(df)
        df = df.replace(to_replace=self.empty_token, value=None)
        return df

    def _add_tokens(self, df, tokenize_columns, tokenizer_name, is_pred=False):
        """
        Map all words in df[tokenize_columns] to a tokenizer. Tokenize the descriptions to max 11 words (longest in
        train set) and return the same dataframe with these 11 tokens added. Descriptions with less than 11 tokens are
        padded with 0.0 till length of 11
        :param df: dataframe to get data to tokenize from
        :param tokenize_columns: the column to tokenize
        :param tokenizer_name: the name for writing the tokenizer to disk
        :return: df: dataframe with the 11 tokens appended
        :return: t: the tokenizer of the df[tokenize_columns] containing mappings
        """
        if is_pred:
            df = df.copy()
            t = pickle.load(open(self.artefact_path + tokenizer_name + '.pickle', 'rb'))
            sequences = t.texts_to_sequences(df[tokenize_columns])

            names = ['x' + str(i) for i in range(4)]
            padded_seq = sequence.pad_sequences(sequences, maxlen=4, padding='post', truncating='post', value=0.0)
            sequences_df = pd.DataFrame(padded_seq, columns=names)
            for col in sequences_df.columns:
                df[col] = sequences_df[col].tolist()
            return df, t
        else:
            t = Tokenizer(oov_token='unknownword')
            t.fit_on_texts(df[tokenize_columns])
            with open(self.artefact_path + tokenizer_name + '.pickle', 'w+b') as handle:
                pickle.dump(t, handle, protocol=pickle.HIGHEST_PROTOCOL)

            sequences = t.texts_to_sequences(df[tokenize_columns])
            names = [f'x{i}' for i in range(4)]
            padded_seq = sequence.pad_sequences(sequences, maxlen=4, padding='post', truncating='post', value=0.0)
            sequences_df = pd.DataFrame(padded_seq, columns=names)
            for col in sequences_df.columns:
                df[col] = sequences_df[col].tolist()
            return df, t

    def _get_label_mapping(self, df=None, is_pred=False):
        """
        Create a consistent mapping of product tags and usable input for the preprocessing. This means the result can be
        expressed as "Champ./Mouss. Wijn" instead of "champ mouss wijn".
        :param df: dataframe of which to map labels
        :return: label_mapping: a dict with the label mapping
        :return: rev_label_mapping: the inverse of the label_mapping
        """

        if is_pred:
            with open(self.artefact_path + 'label_mapping' + '.pickle', 'rb') as handle:
                label_mapping = pickle.load(handle)
        else:
            assert not df.empty
            label_mapping = {}

            d = [df[col].unique().tolist() for col in self.all_flds] + [[self.empty_token]]

            regex = re.compile('[^a-z0-9\\s]')
            for item in itertools.chain(*d):
                if type(item) == str:
                    label_mapping[item] = regex.sub(' ', item.lower())
            with open(self.artefact_path + 'label_mapping' + '.pickle', 'w+b') as f:
                pickle.dump(label_mapping, f, pickle.HIGHEST_PROTOCOL)

        rev_label_mapping = {v: k for k, v in label_mapping.items()}
        return label_mapping, rev_label_mapping

    def _get_label_tokenizer(self, df=None, is_pred=False):
        """
        Create a label tokenizer. Example: "drinks" -> 1
        :param df: dataframe where labels are tokenized from
        :return: the label tokenizer
        """
        if is_pred:
            with open(self.artefact_path + 'label_tokenizer' + '.pickle', 'rb') as handle:
                label_tokenizer = pickle.load(handle)
        else:
            assert not df.empty
            label_tokenizer = Tokenizer(split='€', oov_token='unknownword')

            total_classes = [df.loc[~df[col].isna(), col].unique().tolist() for col in self.all_flds] + [
                [self.empty_token]]
            cleaned_labels = [self.label_mapping.get(x) for sublist in total_classes for x in sublist]
            label_tokenizer.fit_on_texts(cleaned_labels)

            with open(self.artefact_path + 'label_tokenizer' + '.pickle', 'w+b') as handle:
                pickle.dump(label_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        reverse_label_tokenizer = dict(map(reversed, label_tokenizer.word_index.items()))
        return label_tokenizer, reverse_label_tokenizer

    def _remove_columns_not_for_classification(self, df, classification_type):
        """
        Remove columns not used in classification
        :param df: the dataframe to remove columns from
        :param classification_type: what columns to remove depends on what kind of classification we're doing
        :return: a copy of df with only relevant columns
        """
        regex = re.compile(r"x\d")
        default_cols = list(filter(regex.match, df.columns))
        parent_cols = []
        for col in self.all_flds:
            if col == classification_type:
                break
            parent_cols.append(col)

        if classification_type not in self.all_flds[:5]:
            fields = self.all_flds[5:]
            fields.remove(classification_type)
            for x in fields:
                try:
                    parent_cols.remove(x)
                except:
                    logger.debug('Column was not in index')
        return df[parent_cols + default_cols]

    def _get_trainable_rows(self, df, classification, is_pred=False):
        """
        Remove all rows unusable for a specific tag classification
        :param df: df to remove rows from
        :param classification: the type of classification. Options are group, category, subcategory, type and brand
        :return: df with the unusable rows removed
        """
        if is_pred:
            group_drinks = self.label_tokenizer.word_index.get('drinks')
            if classification == 'tag_Group':
                altered_df = df[(df['tag_Group'].isna())]
            elif classification == 'tag_Category':
                altered_df = df[(df['tag_Group'] == group_drinks)
                                & (df['tag_Category'].isna())
                                ]

            elif classification == 'tag_Subcategory':
                altered_df = df[(df['tag_Group'] == group_drinks)
                                & ~(df['tag_Category'].isna())
                                & (df['tag_Subcategory'].isna())
                                ]

            elif classification == 'tag_Type':
                altered_df = df[(df['tag_Group'] == group_drinks)
                                & ~(df['tag_Category'].isna())
                                & ~(df['tag_Subcategory'].isna())
                                & (df['tag_Type'].isna())
                                ]

            elif classification == 'tag_Brand':
                altered_df = df[(df['tag_Group'] == group_drinks)
                                & ~(df['tag_Category'].isna())
                                & ~(df['tag_Subcategory'].isna())
                                & ~(df['tag_Type'].isna())
                                & (df['tag_Brand'].isna())
                                ]
            elif classification == 'tag_Mixer_combination':
                altered_df = df[(df['tag_Group'] == group_drinks)
                                & ~(df['tag_Category'].isna())
                                & ~(df['tag_Subcategory'].isna())
                                & (df['tag_Mixer_combination'].isna())
                                ]
            else:
                altered_df = df[(df['tag_Group'] == group_drinks)
                                & ~(df['tag_Category'].isna())
                                & ~(df['tag_Subcategory'].isna())
                                & (df[classification].isna())
                                ]
        else:

            if classification == 'tag_Group':
                altered_df = df[~(df['tag_Group'].isna())]
            elif classification == 'tag_Category':
                altered_df = df[(df['tag_Group'] == 'Drinks')
                                & ~(df['tag_Category'].isna())]

            elif classification == 'tag_Subcategory':
                altered_df = df[(df['tag_Group'] == 'Drinks')
                                & ~(df['tag_Category'].isna())
                                & ~(df['tag_Subcategory'].isna())]

            elif classification == 'tag_Type':
                altered_df = df[(df['tag_Group'] == 'Drinks')
                                & ~(df['tag_Category'].isna())
                                & ~(df['tag_Subcategory'].isna())
                                & ~(df['tag_Type'].isna())
                                ]

            elif classification == 'tag_Brand':
                altered_df = df[(df['tag_Group'] == 'Drinks')
                                & ~(df['tag_Category'].isna())
                                & ~(df['tag_Subcategory'].isna())
                                & ~(df['tag_Type'].isna())
                                & ~(df['tag_Brand'].isna())
                                ]
            elif classification == 'tag_Mixer_combination':
                altered_df = df[(df['tag_Group'] == 'Drinks')
                                & ~(df['tag_Category'].isna())
                                & ~(df['tag_Subcategory'].isna())
                                & ~(df['tag_Mixer_combination'].isna())
                                ]
            else:
                altered_df = df[(df['tag_Group'] == 'Drinks')
                                & ~(df['tag_Category'].isna())
                                & ~(df['tag_Subcategory'].isna())
                                & ~(df[classification].isna())
                                ]
        return altered_df

    def _label2categorical(self, df, label_column, is_eval=False):
        """
        Convert textual labels to the corresponding categorical values using the label mapping and tokenizer
        example: "Drinks" -> [0, 1, 0, 0]
        :param df: dataframe with data we will convert
        :param label_column: the column of df to convert
        :return: categorical_labels: list of the categorical labels
        """
        if is_eval:
            temp_df = df.copy()
            temp_df[label_column] = df[label_column].map(lambda x: self.label_mapping.get(x))
            temp_df.dropna(subset=[label_column], inplace=True)
            tokenized_labels = self.label_tokenizer.texts_to_sequences(temp_df[label_column])

            # Drop rows with labels not in label_mapping
            temp_df['temp_seq'] = tokenized_labels
            temp_df['temp_seq'] = [x if x else None for x in tokenized_labels]
            temp_df.dropna(subset=['temp_seq'], inplace=True)

            categorical_labels = to_categorical(temp_df['temp_seq'].tolist(),
                                                num_classes=len(self.label_tokenizer.word_index) + 1)
            temp_df.drop('temp_seq', axis=1)
            return categorical_labels, temp_df
        else:
            temp_df = df[[label_column]]
            temp_df[label_column] = df[label_column].map((lambda x: self.label_mapping.get(x)))
            tokenized_labels = self.label_tokenizer.texts_to_sequences(temp_df[label_column])
            categorical_labels = to_categorical(tokenized_labels, num_classes=len(self.label_tokenizer.word_index) + 1)
            return categorical_labels

    def _tag_columns_to_tokens(self, src_df):
        """
        Convert on a per-column basis the labels to numerical using the associated tokenizers for those columns
        :param src_df: the dataframe of which to transform to numerical labels
        :return: the dataframe with all relevant tag columns transformed to numerical values
        """
        df = src_df.copy()
        for fld in self.all_flds:
            if fld in df:
                df[fld] = [self.label_tokenizer.word_index.get(self.label_mapping.get(x)) for x in df[fld]]
        return df

    def _num2labels(self, df, column):
        """
        Convert numerical values back to textual tags
        :param df: dataframe to retrieve values from
        :param column: column to translate to labels
        :return: label translation is done in place
        """
        df[column] = [self.reverse_label_mapping.get(self.reverse_label_tokenizer.get(x)) for x in df[column]]

    @staticmethod
    def _conf_eval(df, eval_df, correct, top_n=100, highest=True):
        """
        Return values with highest confidence, being either correct or incorrect.
        :param df: dataframe with all product information
        :param eval_df: df where we will add all usable information for evaluation (id, description, correctness,
        conf, tag, predicted tag)
        :param correct: Whether you want correct or incorrect values
        :param top_n: number of results you want
        :param highest: Whether you want the highest or lowest confidences
        :return:
        """
        assert eval_df.index.isin(df.index).all
        requested_results = eval_df[(eval_df.actual_classes == eval_df.pred_classes) == correct]
        requested_results[['id', 'description']] = df[['id', 'description']]
        requested_results = requested_results[['id', 'description', 'actual_classes', 'pred_classes', 'confidence', ]]
        end_result = requested_results.sort_values('confidence', ascending=not highest)[:top_n]
        return end_result

    @staticmethod
    def _fix_column_order_and_dtypes(df):
        c = df.columns.tolist()
        start_columns = ['id', 'description', 'unit_price', 'venue_id', 'tag_Container', 'tag_Volume', 'tag_Group',
                         'tag_Group_pred', 'tag_Group_pred_conf', 'tag_Category', 'tag_Category_pred',
                         'tag_Category_pred_conf', 'tag_Subcategory', 'tag_Subcategory_pred',
                         'tag_Subcategory_pred_conf', 'tag_Type', 'tag_Type_pred', 'tag_Type_pred_conf', 'tag_Brand',
                         'tag_Brand_pred', 'tag_Brand_pred_conf']
        logger.debug(f'beginning of column order:\n {start_columns}')
        for col in start_columns:
            try:
                c.remove(col)
            except Exception as e:
                logger.info(e)
                logger.info(col)
        try:
            df_new_columns = df[start_columns + c]
        except:
            df_new_columns = df
        fix_dtype_columns = ['product_id', 'unit_price', 'venue_id', 'id']
        df_new_columns[fix_dtype_columns] = df_new_columns[fix_dtype_columns] \
            .fillna(-1) \
            .astype(int) \
            .astype(str) \
            .replace('-1', np.nan)
        return df_new_columns

    @classmethod
    def _preprocess_data(cls, df, empty_token, mode):
        """
        Preprocess the data
        :param df: dataframe to preprocess
        :return: preprocessed dataframe
        """
        if mode == 'predict':
            df = df.dropna(subset=['description'])
            df = df[df.tag_Group.isna()]
        elif mode == 'train':
            df = df.fillna({'tag_Category': empty_token,
                            'tag_Subcategory': empty_token,
                            'tag_Type': empty_token,
                            'tag_Brand': empty_token,
                            'tag_Mixer_combination': empty_token,
                            'tag_SKU': empty_token})
            df = df.dropna(subset=['description', 'tag_Group'])
            df = df[df['unit_price'] < 500000]
            df = df[~df.venue_id.isin(cls.excluded_venues)]
        elif mode == 'evaluate':
            df = df.dropna(subset=['description'])

        df = df[(df['id'] != 9999999) & (df['product_id'] != -1)]
        return df

    @staticmethod
    def _fill_na_cells_with_pred(target_df, source_df, classification_type):
        """
        Fill NA values in the tag columns with predicted values
        :param target_df: df to insert column into
        :param source_df: df to retrieve information from
        :param classification_type: type of classification that is being done
        :return: column assignment is done by reference. Nothing is returned
        """
        try:
            if classification_type == 'tag_Group':
                pass
            elif classification_type == 'tag_Category':
                target_df['tag_Group'] = source_df['tag_Group'].fillna(source_df['tag_Group_pred'])
            elif classification_type == 'tag_Subcategory':
                target_df['tag_Group'] = source_df['tag_Group'].fillna(source_df['tag_Group_pred'])
                target_df['tag_Category'] = source_df['tag_Category'].fillna(source_df['tag_Category_pred'])
            elif classification_type == 'tag_Type':
                target_df['tag_Group'] = source_df['tag_Group'].fillna(source_df['tag_Group_pred'])
                target_df['tag_Category'] = source_df['tag_Category'].fillna(source_df['tag_Category_pred'])
                target_df['tag_Subcategory'] = source_df['tag_Subcategory'].fillna(source_df['tag_Subcategory_pred'])
            elif classification_type == 'tag_Brand':
                target_df['tag_Group'] = source_df['tag_Group'].fillna(source_df['tag_Group_pred'])
                target_df['tag_Category'] = source_df['tag_Category'].fillna(source_df['tag_Category_pred'])
                target_df['tag_Subcategory'] = source_df['tag_Subcategory'].fillna(source_df['tag_Subcategory_pred'])
                target_df['tag_Type'] = source_df['tag_Type'].fillna(source_df['tag_Type_pred'])
            else:
                target_df['tag_Group'] = source_df['tag_Group'].fillna(source_df['tag_Group_pred'])
                target_df['tag_Category'] = source_df['tag_Category'].fillna(source_df['tag_Category_pred'])
                target_df['tag_Subcategory'] = source_df['tag_Subcategory'].fillna(source_df['tag_Subcategory_pred'])
                target_df['tag_Type'] = source_df['tag_Type'].fillna(source_df['tag_Type_pred'])
                target_df['tag_Brand'] = source_df['tag_Brand'].fillna(source_df['tag_Brand_pred'])
        except:
            print('some columns doesnt exist. Please run all colums')


def cnn_model(input_length, vocab_size, num_class, class_type):
    """
    The model definition
    :param input_length: the length of the feature input
    :param vocab_size: number of different words in the vocabulary (unique words)
    :param num_class: number of classes the model will predict
    :param class_type: classification type or column that is being predicted. Use different nr. nodes for dense layers
    for different classifications.
    :return:
    """
    input_1 = Input(shape=(input_length,))
    x = Embedding(vocab_size, 64, input_length=input_length, name='embedding')(input_1)
    x = SpatialDropout1D(0.3)(x)
    x = Conv1D(256, kernel_size=3, activation='relu', name='1dconv')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    x = Dropout(0.4)(x)
    desc_embeds = x

    # We increase the number of dense nodes when prediction things with more classes. This improves the expressiveness
    # of the model and improves performance.
    # used for tag_type
    if class_type == 'tag_Type':
        x = Dense(256, activation='relu')(desc_embeds)
        x = Dropout(0.5, seed=5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5, seed=5)(x)
    # used for tag_Brand
    elif class_type == 'tag_Brand':
        x = Dense(512, activation='relu')(desc_embeds)
        x = Dropout(0.5, seed=5)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5, seed=5)(x)
    # used for anything that is not tag_Type or tag_Brand
    else:
        x = Dense(128, activation='relu')(desc_embeds)
        x = Dropout(0.4, seed=5)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.4, seed=5)(x)

    predictions = Dense(num_class, activation='softmax')(x)

    model = Model(inputs=[input_1], outputs=predictions)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    return model


def fit_command(path, trainflds=None, nrsamples=0):
    if trainflds is None:
        trainflds = []
    else:
        trainflds = [x.strip() for x in trainflds.split(',')]
    product_model = ProductModel(path, from_path=True)
    product_model.fit(train_flds=trainflds, verbose=1, nr_samples=nrsamples)


def predict_command(path, pred_flds=None, nrsamples=0):
    if pred_flds is None:
        pred_flds = []
    else:
        pred_flds = [x.strip() for x in pred_flds.split(',')]
    product_model = ProductModel(path, from_path=True)
    predictions = product_model.predict(pred_flds=pred_flds, nr_samples=nrsamples)
    predictions.to_csv('data_predicted.csv', sep=';', header=predictions.keys(), index=False)


def evaluate_command(path, eval_flds=None, nrsamples=0, plot=False):
    if eval_flds is None:
        eval_flds = []
    else:
        eval_flds = [x.strip() for x in eval_flds.split(',')]
    product_model = ProductModel(path, from_path=True)
    product_model.evaluate(eval_flds=eval_flds, nr_samples=nrsamples, plot=plot)


if __name__ == "__main__":
    run()
