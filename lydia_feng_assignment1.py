'''
Lydia Feng
CS-584: Applied Biomedical Natural Language Processing
Abeed Sarker
Assignment 1
'''

import re
import nltk
from nltk.tokenize import sent_tokenize
import pandas as pd
import numpy as np
import itertools
import Levenshtein
import xlsxwriter


# read in posts and symptom lexicon
textpath = './s11.xlsx'
symptompath = './COVID-Twitter-Symptom-Lexicon.txt'
data = pd.read_excel(textpath)
file = pd.DataFrame(data, columns=['ID', 'DATE', 'TEXT'])


# create result file
wb = xlsxwriter.Workbook('results.xlsx')
results = wb.add_worksheet()
results.write(0, 0, "ID")
results.write(0, 1, "Symptom CUIs")
results.write(0, 2, "Negation Flag")


# create negation list
inneg = open('./neg_trigs.txt')
negtext = inneg.read()
neglist = list(negtext.split("\n"))


# create symptom dictionary
symptom_dict = {}
with open(symptompath) as f:
    for line in f:
        linesplit = line.strip().split("\t")
        symptom_dict[linesplit[2]] = linesplit[1]

#include title of CUI as another key to match
simple_symptom_dict = {}
with open(symptompath) as f:
    for line in f:
        linesplit = line.strip().split("\t")
        symptom_dict[linesplit[0].lower()] = linesplit[1]
symptom_dict.update(simple_symptom_dict)


# create and move word window for fuzzy matching
def run_window(words, window_size):
    word_iterator = iter(words)
    word_window = tuple(itertools.islice(word_iterator, window_size))
    yield word_window
    for w in word_iterator:
        word_window = word_window[1:] + (w,)
        yield word_window


# threshhold for Levenshtein similarity depends on length of key to match
def determine_thresh(leng):
    if (leng == 1):
        return 0.95
    if (leng == 2):
        return 0.85
    else:
        return 0.8


# check if match is within scope of a negation trigger
def in_scope(neg_end, sentences, symptom):
    negated = False
    followingtext = str(sentences[neg_end:])
    scope = ' '.join(followingtext[:min(len(followingtext), len(symptom.split()) + 3)])

    # if there's a match within 3 terms of a neg, check if there's a sentence end in scope
    match_object = re.search(symptom, scope)
    if match_object:
        period = re.search('\.', scope)
        next_negation = 1000
        for neg in neglist:
            negmatch = re.search(neg, followingtext)
            if negmatch:
                index = followingtext.find(neg)
                if index < next_negation:
                    next_negation = index
        if period:
            if (period.start() > match_object.start()) & (next_negation > match_object.start()):
                negated = True
        else:
            negated = True
    return negated


# iterate through posts
for i in range(len(file)):
    text = ""
    symptoms = []  # list to hold total matches per post

    text = str(file.loc[i, 'TEXT'])
    sentences = sent_tokenize(text.lower())

    # for each sentence and for each symptom, perform exact and fuzzy matching
    for sent in sentences:
        # sentence level lists to maintain correct ordering of char indices
        sentence_symptoms = []
        sentence_fuzzy_symptoms = []
        sentence_combined_symptoms = []
        for key in symptom_dict:

            # exact matching
            match_obj = re.search(key, sent)
            if match_obj:
                # symptoms[3] is negation marker where default is not negated
                sentence_symptoms.append((match_obj.start(), match_obj.group(), symptom_dict[key], 0))

            # fuzzy matching
            thresh = determine_thresh(len(key.split()))
            max_sim = -1
            sim_score = 0
            best_match = ""

            for window in run_window(sent.split(' '), len(key.split())):
                window_string = ' '.join(window)
                sim_score = Levenshtein.ratio(window_string, key)
                if (sim_score >= thresh) & (sim_score > max_sim):
                    max_sim = sim_score
                    best_match = window_string
            if best_match != "":
                # fuzz_symptoms[3] is negation marker where default is not negated
                sentence_fuzzy_symptoms.append((sent.find(best_match), best_match, symptom_dict[key], 0))

            # remove repeated annotations within 8 chars of each other that have same CUI
            for tup1 in sentence_fuzzy_symptoms:
                for tup2 in sentence_fuzzy_symptoms:
                    if (tup1[2] == tup2[2]) & (abs(tup1[0] - tup2[0]) < 8) & (tup1 != tup2):
                        if tup1[3] >= tup2[3]:
                            sentence_fuzzy_symptoms.remove(tup2)
                        else:
                            sentence_fuzzy_symptoms.remove(tup1)

        # remove fuzzy matches that start at same index as an exact match, or are within 8 chars of an exact match with the same CUI
        for j in range(len(sentence_fuzzy_symptoms) - 1, -1, -1):
            for symp in sentence_symptoms:
                if (sentence_fuzzy_symptoms[j][0] == symp[0]) | (
                        (abs(sentence_fuzzy_symptoms[j][0] - symp[0]) < 8) & (sentence_fuzzy_symptoms[j][2] == symp[2])):
                    del sentence_fuzzy_symptoms[j]
                    break


        # combine fuzzy and exact matches
        sentence_combined_symptoms = sentence_symptoms + sentence_fuzzy_symptoms


        # check if matches are within scope of negation trigger
        for match in sentence_combined_symptoms:
            is_negated = False
            for neg in neglist:

                # find negation triggers in text and see if there is a match within the scope of one
                for neg_match in re.finditer(r'\b' + neg + r'\b', text):
                    is_negated = in_scope(neg_match.end(), sent, re.sub(r'[()+*]', '', match[1]))
                    if is_negated:
                        match[3] = 1
                        break

        # sort symptoms within sentence level
        sentence_combined_symptoms.sort(key=lambda tup: tup[0])

        # add sentence-level symptoms to overall post symptoms
        symptoms = symptoms + sentence_combined_symptoms

    # write CUI and negation flag of matches into excel
    cui_string = ""
    neg_string = ""

    for tup in symptoms:
        cui_string = cui_string + "$$$" + tup[2]
        neg_string = neg_string + "$$$" + str(tup[3])
    cui_string = cui_string + "$$$"
    neg_string = neg_string + "$$$"

    results.write(i + 1, 0, file.loc[i, 'ID'])
    results.write(i + 1, 1, cui_string)
    results.write(i + 1, 2, neg_string)


wb.close()
