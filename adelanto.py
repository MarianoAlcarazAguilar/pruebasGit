#! PATH/TO/ENVIRONMENT/BIN

import os
import re
import gzip
import magic
import string
import numpy as np
import pandas as pd
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy import spatial

class WordRelate:
    """
    This class implements all the necessary functionality
    to perform similarity analysis between words using distributed representations.
    """

    def __init__(self, home_dir='.'):
        self.home_dir = home_dir
        self.data_path = os.path.join(self.home_dir, 'data', 'text_comp', 'texts')
        self.fig_path = os.path.join(self.home_dir, 'figs')
        # -----------------------------------
        # We'll store all our results at a collection based level i.e.
        # we are going to access every structure within the class
        # using the collection id as keys.
        # TODO
        # Initialize the values (empty dics) for
        # - voc
        # - ivoc
        # - collections DONE
        # - vrm
        # - reduced_vrm
        # (5 points)
        # -----------------------------------

        # Your code goes here (~ 1 - 5 lines)
        self.voc = {}
        self.ivoc = {}
        self.collections = {}
        self.vrm = {}
        self.reduced_vrm = {}

        print(f"Files found in storage: \n {os.listdir(self.data_path)}")


    @staticmethod
    def proc_line(line):
        """
        This function parses each input line:
        - sets every word into lowercase
        - remove non-words (except single white space)
        - remove empty lines
        - returns a list of words.
        param line: the line to process
        :return: processed line
        """
        # -----------------------------------
        # TODO
        # complete the parsers, currently, each word
        # line is output as it is.
        # (10 points)
        # -----------------------------------

        #Esto ya debe recibir la línea en lower
        return re.sub(f"[{string.punctuation}]+", "", line).split()




    def get_voc(self, collection_id, sw, top_freq_words=2000):
        """
        Get a series with the most common words in the collection.

        param collection_id: the id of the collection to process
        :param sw: list of stop words
        param freq_words: maximum number of words to include in output
        :return:
            - self.voc: a series containing the vocabulary: {word: index},

            - self.ivoc: a series containing the vocabulary: {index: word} (this will be handy for word retrieval)
        """
        # -----------------------------------
        # TODO
        # Parse the texts in the collection and
        # build a series with the top_freq_words.
        # Sort the index of voc based on the words.
        # Sort the index of ivoc based on the numerical value.
        # (10 points)
        # -----------------------------------

        # Your code goes here (~ 2 lines)
        self.read_collection(collection_id)
        # Primero buscamos todas las palabras y contamos cuantas veces aparecen
        # Luego ordenamos por número de apraciones
        # Luego les damos el indice por orden de apraciones
        # Nos quedamos solo con las primeras 2000 palabras

        aux_dict = {}
        # Sacamos la lista de libros del id
        for book in self.collections[collection_id]:
            for line in book:
                for word in line:
                    if word not in sw:
                        try:
                            aux_dict[word] += 1
                        except:
                            aux_dict[word] = 1


        sort_orders = sorted(aux_dict.items(), key=lambda x: x[1], reverse=True)
        # Agregamos a un diccionario las primeras 10 palabras
        minSize = min(top_freq_words, len(sort_orders))
        top_words = {}
        other_top_words = {}
        for i in range(minSize):
            top_words[i] = sort_orders[i][0]
            other_top_words[sort_orders[i][0]] = i

        # Make order monotonic to improve performance.
        self.voc[collection_id] =  other_top_words # Your code goes here
        # Get inverse index for word vocs
        self.ivoc[collection_id] = top_words # Your code goes here
        print(f'Monotonic index:{self.voc[collection_id].index.is_monotonic}')


    def dist_rep(self, collection_id, ws=4):
        """
        In this function we get the distributed representation of words based on
        cooccurrence along windows of size ws. This is the most
        challenging question in the project.
        Check https://aclanthology.org/W14-1503.pdf for best practices on this methodology.
        param collection_id:
        :param ws: size of the context window
        :return:
         - self.vrm: a vectorized representation of the words
        """
        voc_size = len(self.voc[collection_id])
        self.vrm[collection_id] = np.zeros((voc_size, voc_size))
        for text in self.collections[collection_id]:
            text = np.array([w for w in text if w in self.voc[collection_id]], dtype=object)
            # -----------------------------------
            # TODO
            # For each word in text, build a window
            # of size 2*ws. Containing the vocabulary indexes of ws words prior to the word
            # of interest and ws words after. For example
            # test = [START this is very a simple example]
            # w = this
            # the output window should contain the indexes of the words
            # START, is, very, a, simple.
            # Be sure to structure your output as follows:
            # (iw_c, (iw_in1, iw_in2, ..., iw_inN), (cw_in1, cw_in2, ..., cw_inN))
            # where
            # - iw_c: is the voc index of the central word of the window
            # - iw_ini: is the voc index of the i'th word in the window
            # - cw_ini: is the number of times w_ini appears in the window
            # (35 points)
            # Compare your results with the precomputed matrices at:
            # ./home_dir/data/tests
            # np.array_equal
            # -----------------------------------

            # Your code goes here ( 1 ~ 10 lines)

            for word, (related, values) in indexes:
                # -----------------------------------
                # TODO
                # Update the count values in self.vrm.
                # (10 points)
                # -----------------------------------

                # Your code goes here (1 line)


    def ppmi_reweight(self, collection_id):
        """
        In this section we apply ppmi transformation to vrm
        """
        # -----------------------------------
        # TODO
        # Compute the matrix of expected counts to
        # update vrm.
        # expected_i_j = (row_sum_i*col_sum_j)/tot_sum
        # (15 points)
        # -----------------------------------

        # Your code goes here (~ 1 - 4 lines)

        with np.errstate(divide='ignore'):
            log_vals = np.log(self.vrm[collection_id]/expected)
        self.vrm[collection_id] = np.maximum(log_vals, 0)


    def dim_redux(self, collection_id, dim_reducer='pca'):
        """
        Apply dimensionality reduction
        :return:
        """
        if dim_reducer == 'tsne':
            self.reduced_vrm[collection_id] = TSNE(n_components=2,
                                                   init='random').fit_transform(self.vrm[collection_id])
        if dim_reducer == 'pca':
            self.reduced_vrm[collection_id] = PCA(n_components=2).fit_transform(self.vrm[collection_id])


    def plot_reps(self):
        """
        Plots the reduced representations computed for each collection
        :return:
        """

        if not os.path.isdir(self.fig_path):
            print('Making figure directory')
            os.mkdir(self.fig_path)

        for i, collection in enumerate(self.collections.keys()):
            fig, ax = plt.subplots(figsize=(12, 10))
            x = self.reduced_vrm[collection][:, 0]
            y = self.reduced_vrm[collection][:, 1]
            ax.scatter(x, y)
            for i, txt in enumerate(self.ivoc[collection]):
                ax.annotate(txt, (x[i], y[i]))
            print(f'saving figure {collection}')
            fig.savefig(os.path.join(self.fig_path, f'{collection}.png'))


    def make_bulk_collections(self):
        """
        Unifies all collections in one single collection with key 'BULK'
        :return:
        """
        bulk = []
        for i, collection in enumerate(self.collections.keys()):
                bulk += self.collections[collection]
        self.collections['BULK'] = bulk


    def get_word_relatedness(self, collection):
        """

        param word_relate_path:
        :return:
        """
        global annotated_sim, predicted_sim
        word_relate_path = os.path.join(self.home_dir, 'data','relatedness', 'wr.csv')
        wr = pd.read_csv(word_relate_path)
        for index, row in wr.iterrows():
            w1 = row.word1
            w2 = row.word2
            s  = row.score
            # Check if words are in voc
            annotated_sim = []
            predicted_sim = []
            print(w1)
            print(w2)
            if (w1 in self.voc[collection]) and (w2 in self.voc[collection]):
                annotated_sim.append(s)
                similarity = spatial.distance.cosine(wc.vrm[collection][wc.voc[collection][w2]],
                                                     wc.vrm[collection][wc.voc[collection][w1]])
                predicted_sim.append(similarity)
        correlation = np.corrcoef(np.array(annotated_sim), np.array(predicted_sim))[0, 1]
        print(f'Found {len(predicted_sim)} words to relate in voc. Pearson Correlation: {correlation}')
        return correlation

    def read_collection(self, collection_id):
        collection_path = os.path.join(self.data_path, collection_id)
        # Read each file in collection
        texts = []
        for file in os.listdir(collection_path):
            file_path = os.path.join(collection_path, file, 'main_text.txt')
            # Take care of gzipped files
            if re.match(r'^gzip', magic.from_file(file_path)):
                with gzip.open(file_path, 'rb') as fa:
                    lines = fa.read()
            else:
                with open(file_path, encoding="utf-8") as g:
                    # Read lines and process them. (notice we are removing empty lines)
                    lines = g.readlines()
            # -----------------------------------
            # TODO
            # - Call proc line on each line of input
            # - Make sure to add 'START' and 'END' tokens to the start and end of each sentence.
            # - Preserve only non empty lines.
            # (15 points)
            # -----------------------------------

            # Your code goes here (~ 1 - 3 lines)
            lines = [re.sub(r"\n", "", x).lower() for x in lines]
            lines = [f"START {x} END" for x in lines]
            lines = [self.proc_line(x) for x in lines if len(self.proc_line(x)) > 2]
            texts.append(lines)
        # Add texts to the collections.
        # Texts es una lista que tiene listas (Cada una corresponde a un texto) y cada lista tiene listas con las
        # palabras de cada línea
        self.collections[collection_id] = texts


if __name__ == '__main__':
    # Check available collections
    collections = os.listdir(os.path.join('data', 'text_comp', 'texts'))
    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument('--collection',
                        choices=collections)
    args = parser.parse_args()
    # Read stopwords
    with open('./data/stop_words/stopwords.txt') as f:
        stopwords = f.read().split('\n')
    # Instantiate WordRelate object
    wc = WordRelate()
    # Read collection argument
    collection = args.collection

    # -----------------------------------
    # TO DO
    # -----------------------------------
    # 1.- read collection 00
    # 2.- get vocabularies
    # 3.- generate distributed representations
    # 4.- apply ppmi
    # 5.- apply dimensionality reduction
    # 6.- Plot results
    # To test your execution run:
    # ./wordrelatedness --collection '[collection_id]'
    # Your final output should produce two plots
    # such as the one displayed in:
    # ./home_dir/figs/.
    # -----------------------------------

    # Your code goes here (~ 7 lines)
