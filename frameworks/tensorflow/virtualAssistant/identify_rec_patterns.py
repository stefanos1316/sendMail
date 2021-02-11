import os
import pandas as pd
import spacy
import re
from spacy.matcher import Matcher
from sklearn.metrics import precision_recall_fscore_support
from spacy.attrs import LOWER, POS, ENT_TYPE, LEMMA
from spacy.tokens import Doc
import numpy

rec_target_noun = ["project","api","tutorial","library","example","book","resource","elm","python","clojure",
                    "way","practice","approach","strategy","technique","explanation","choice", "trick","tip",
                    "alternative","recommendation","opinion", "boilerplate","service","framework","project","package",
                    "suggestion","linter","procedure"]
action_verbs = ["have", "know","use","try"]
rec_verbs = ["point","recommend","suggest","advise","tell"]
rec_positive_adjective = ["good","better","best","right","optimal","ideal","common","established","popular","great",
                          "nice","nicer","elegant","efficient","easy","easier","simple","simpler","simplistic",
                          "clean","cleaner","pro","other","idiomatic","pythonic","clojurian","clojure","clojury","elm",
                          "elmish","generally","accepted","currently","most","normal","ok","appropriate","standard"]
all_caps_noun_regex = "[A-Z]{2,}s?"
vp_pattern = [{"POS": {"NOT_IN": ["VERB","AUX"]}, "OP": "*"},
              {"POS": {"IN": ["VERB","AUX"]}}]
np_pattern = [{"POS": {"NOT_IN": ["NOUN","PROPN"]}, "OP": "*"},
              {"POS": {"IN": ["NOUN","PROPN"]}}]

# P_WHAT_RECADJ_TARGETNOUN
# Question with postitive adjective and noun
# What|Which [VERB_TO_BE] [ADJECTIVE]* <rec_target_noun>
def setup_p_what_recadj_targetnoun(matcher):
    base_pattern = [{"LEMMA": {"IN": ["what","which"]}},
                    {"LOWER": {"IN": ["would","one"]},"OP": "?"},
                    {"LEMMA": "be", "OP": "?"},
                    {"POS": "DET","OP": "?"},
                    {"POS": "ADJ","OP": "?"}]
    what_is_pattern1 = base_pattern + [
                    {"LEMMA": {"IN": rec_target_noun}}] + vp_pattern
    what_is_pattern2 = base_pattern + [
                    {"TEXT": {"REGEX": all_caps_noun_regex}}] + vp_pattern
    matcher.add("P_WHAT_RECADJ_TARGETNOUN",
                on_match=on_match_p_what_recadj_targetnoun,
                patterns=[what_is_pattern1, what_is_pattern2])

def on_match_p_what_recadj_targetnoun(matcher, doc, id, matches):
    #print('P_WHAT_RECADJ_TARGETNOUN Matched!', doc.text)
    pass


# P_COULD_ANYONE
# Query about the existence of a person with knowledge
# Could|Can anyone|someone rec_verb [me|us] [to] [an] <adjective>* <rec_target_noun>
def setup_p_could_anyone(matcher):
    could_pattern = [{"LOWER": {"IN": ["can", "could"]}},
                     {"LOWER": {"IN": ["someone", "anyone"]}},
                     {"LOWER": {"IN": rec_verbs}}]
    matcher.add("P_COULD_ANYONE", on_match_p_could_anyone, could_pattern)

def on_match_p_could_anyone(matcher, doc, id, matches):
    #print('P_COULD_ANYONE Matched!', doc.text)
    pass


# P_ARE_THERE
# Query about the existence an artifact
# Is|Are there any|a|an|some <rec_positive_adjective> <rec_target_noun>
def setup_p_are_there(matcher):
    at_pattern_base = [{"LOWER": {"IN": ["are","is"]}},
                     {"LOWER": "there"},
                     {"LOWER": {"IN": ["any", "a", "some", "an"]}},
                     {"LOWER": {"IN": rec_positive_adjective}, "OP": "+"}]
    # at_pattern1 = at_pattern_base + [{"LEMMA": {"IN": rec_target_noun}}] + vp_pattern
    # at_pattern2 = at_pattern_base + [{"TEXT": {"REGEX": all_caps_noun_regex}}] + vp_pattern
    matcher.add("P_ARE_THERE",
                on_match = on_match_p_are_there,
                patterns = [at_pattern_base])

def on_match_p_are_there(matcher, doc, id, matches):
    #print('P_ARE_THERE Matched!', doc.text)
    pass


# P_IS_IT
def setup_p_is_it(matcher):
    is_it_pattern = [{"LOWER": "is"},
                     {"LOWER": "it"},
                     {"LOWER": {"IN": rec_positive_adjective}, "OP": "+"}]
    matcher.add("P_IS_IT",
                on_match = on_match_p_is_it,
                patterns = [is_it_pattern])


def on_match_p_is_it(matcher, doc, id, matches):
    pass


# P_DOES_ANYONE
# Query about someone having specific knowledge
# [Do|Does] we|anyone|someone <action_verbs> [...] <rec_target_noun>
def setup_p_does_anyone(matcher):
    does_pattern = [{"LOWER": {"IN": ["does", "do"]}, "OP": "?"},
                    {"LOWER": {"IN": ["we", "anyone", "someone"]}},
                    {"LEMMA": {"IN": action_verbs}},
                    {"LEMMA": {"NOT_IN": rec_target_noun}, "OP": "*"},
                    {"LEMMA": {"IN": rec_target_noun}, "OP": "+"}]
    matcher.add("P_DOES_ANYONE", on_match_p_does_anyone, does_pattern)

def on_match_p_does_anyone(matcher, doc, id, matches):
    #print('P_DOES_ANYONE Matched!', doc.text)
    pass


# P_HAS_ANYONE
# Query about someone having specific knowledge
# Has|Have anyone|someone|you <action_verbs> <noun_phrase>
def setup_p_has_anyone(matcher):
    has_pattern = [{"LOWER": {"IN": ["has", "have"]}},
                    {"LOWER": {"IN": ["you", "anyone", "someone"]}},
                    {"LEMMA": {"IN": action_verbs}}] + \
                    np_pattern
    matcher.add("P_DOES_ANYONE", on_match_p_has_anyone, has_pattern)

def on_match_p_has_anyone(matcher, doc, id, matches):
    #print('P_HAS_ANYONE Matched!', doc.text)
    pass


# P_ANY
# Question starting with any and followed by a rec target
# Any <rec_target_noun> <noun_phrase>
def setup_p_any(matcher):
    base_pattern = [{"LOWER": "any"},
                    {"LOWER": {"IN": rec_positive_adjective}, "OP": "+"}]
    vp_narrow_pattern = [{"POS": {"NOT_IN": ["VERB"]}, "OP": "*"},
                         {"POS": "VERB"}]
    #not_i_pattern = [{"TEXT": {"NOT_IN": ["I"]}}]
    any_pattern1 = base_pattern + [{"LEMMA": {"IN": rec_target_noun}}] + vp_narrow_pattern
    any_pattern2 = base_pattern + [{"TEXT": {"REGEX": all_caps_noun_regex}}] + vp_narrow_pattern
    any_pattern3 = [{"LOWER": "any"}, {"LEMMA": {"IN": ["recommend","recommendation","advice","suggestion","alternative","direction"]}}]
    matcher.add("P_ANY", on_match=on_match_p_any, patterns=[any_pattern1, any_pattern2, any_pattern3])

def on_match_p_any(matcher, doc, id, matches):
    #print('P_ANY Matched!', doc.text)
    pass


# P_WHERE_FIND
# Question about the location of a rec item
# Where can|could I|someone|anyone find <rec_target_noun>
def setup_p_where(matcher):
    end_pattern =  [{"LOWER": "find"},
                    {"LEMMA": {"NOT_IN": rec_target_noun}, "OP": "*"},
                    {"LEMMA": {"IN": rec_target_noun}, "OP": "+"}]
    where_pattern1 = [{"LOWER": "where"},
                    {"LOWER": {"IN": ["can","could"]}},
                    {"LOWER": {"IN": ["i", "anyone", "someone"]}},
                    {"LOWER": "find"}] + end_pattern
    where_pattern2 = [{"LOWER": "where"},
                    {"LOWER": "i"},
                    {"LOWER": {"IN": ["can","could"]}}] + end_pattern
    matcher.add("P_ANY", on_match=on_match_p_where, patterns=[where_pattern1, where_pattern2])

def on_match_p_where(matcher, doc, id, matches):
    #print('P_WHERE Matched!', doc.text)
    pass


# P_SHOULD_I_OR
def setup_p_should_i_or(matcher):
    or_pattern1 = [{"LOWER": {"IN": ["should"]}},
                    {"LOWER": {"IN": ["i"]}}] + \
                    np_pattern + \
                  [{"LOWER": {"NOT_IN": ["or"]}, "OP":"*"},
                   {"LOWER": {"IN": ["or"]}}]
    or_pattern2 = [{"LOWER": {"IN": ["or"]}},
                   {"LOWER": {"IN": ["should"]}},
                   {"LOWER": {"IN": ["i"]}}]

    matcher.add("P_DOES_ANYONE", on_match=on_match_p_should_i_or, patterns=[or_pattern1, or_pattern2])

def on_match_p_should_i_or(matcher, doc, id, matches):
    pass


# keeps only utterances that initiate a conversation and
# belong to a single speaker
def filter_potential_questions(all_uter):
    mask = []
    last_thread = -1
    same_speaker = "None"
    for index, row in all_uter.iterrows():
        if row["thread"] != last_thread:
            last_thread = row["thread"]
            same_speaker = row["speaker"]
            mask.append(True)
        elif row["speaker"] == same_speaker:
            mask.append(True)
        else:
            same_speaker = "None"
            mask.append(False)
    return all_uter[mask]


def remove_punct_from_doc(doc):
    indexes = []
    for index, token in enumerate(doc):
        if (token.pos_  in ('PUNCT', 'NUM', 'SYM')):
            indexes.append(index)
    np_array = doc.to_array([LOWER, POS, ENT_TYPE, LEMMA])
    np_array = numpy.delete(np_array, indexes, axis = 0)
    doc2 = Doc(doc.vocab, words=[t.text for i, t in enumerate(doc) if i not in indexes])
    doc2.from_array([LOWER, POS, ENT_TYPE, LEMMA], np_array)
    return doc2


def print_err_diagnostics(data, sent_matches, index):
    if ("is_rec_question" in data.columns):
        if sent_matches:
            if (data["is_rec_question"][index] == 0):
                print("False Positive,\"" + str(data["thread"][index]) + ": " + data["message"][index])
                #print("False Positive,\"" + data["message"][index] + "\"")
        else:
            if (data["is_rec_question"][index] == 1):
                print("False Negative,\"" + data["message"][index] + "\"")


def process_matches(q_data, matcher):
    uter_rec_q = []
    for index, row in q_data.iterrows():
        uter = row['message']
        thread = row['thread']

        #filter out utterances with multiline code (containing triple quote)
        if (not isinstance(uter, str)) or '```' in uter:
            uter_rec_q.append(0)
            continue

        sent_matches = False
        matches = []
        for sentence in nlp(uter).sents:
            sent_nlp = nlp(sentence.text)
            doc = remove_punct_from_doc(sent_nlp)
            matches = matcher(doc)
            if (len(matches) >= 1):
                sent_matches = True
        print_err_diagnostics(q_data, sent_matches, index)

        if sent_matches:
            uter_rec_q.append(thread)
        else:
            uter_rec_q.append(0)

    return uter_rec_q


def run_patterns_dev_set(matcher):
    data = pd.read_csv('../../../data/question/test_set.csv')
    q_data = filter_potential_questions(data)

    print("Total utterances:", len(data.index))
    print("Total convos:", len(data["thread"].unique()))
    print("Total rec questions:", len(data.loc[data['is_rec_question'] == 1]))

    uter_rec_q = process_matches(q_data, matcher)

    assert(len(uter_rec_q) == len(q_data['is_rec_question']))
    precision, recall, fscore, support = \
        precision_recall_fscore_support(q_data['is_rec_question'], [int(u != 0) for u in uter_rec_q], average='binary')

    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))


def run_patterns_test_set_single(chatFileName, matcher):
    data = pd.read_csv(chatFileName)

    print("Total utterances:", len(data.index))
    print("Total convos:", len(data["thread"].unique()))

    q_data = filter_potential_questions(data)
    uter_rec_q = process_matches(q_data, matcher)
    return data.loc[data['thread'].isin(uter_rec_q)]


def run_patterns_test_set(matcher):
    # generating train set with conversations with rec-asking questions
    rec_threads = 0

    outfile = 'outfile.csv'
    if os.path.exists(outfile):
        os.remove(outfile)

    walkdir = "../../data/irc"
    for root, dirnames, filenames in os.walk(walkdir):
        for filename in filenames:
            if not filename.endswith('.csv'):
                continue
            else:
                chatFileName = os.path.join(root, filename)
                print(chatFileName)

                out_df = run_patterns_test_set_single(chatFileName, matcher)
                ths = len(out_df.thread.unique())
                print("Conversations with rec-asking questions:", ths)
                rec_threads = rec_threads + ths

                #write a new csv file containing only the conversations which start with rec-asking question
                with open(outfile, 'a') as fpointer:
                    #local_path = [chatFileName] * len(df.index)
                    #df['FileName'] = local_path
                    out_df['thread'] = out_df['thread'].apply(lambda x: str(x) + '_' + str(root)[11:].replace("/","_"))
                    assert ths == len(out_df.thread.unique())
                    out_df.to_csv(fpointer, header=False, index=False)
    print("Overall conversations with rec-asking questions:", rec_threads)


if __name__ == '__main__':
    nlp = spacy.load("en_core_web_sm")
    matcher = Matcher(nlp.vocab, validate=True)
    setup_p_what_recadj_targetnoun(matcher)
    setup_p_could_anyone(matcher)
    setup_p_are_there(matcher)
    setup_p_any(matcher)
    setup_p_does_anyone(matcher)
    setup_p_has_anyone(matcher)
    setup_p_where(matcher)
    setup_p_should_i_or(matcher)
    setup_p_is_it(matcher)
    #setup_p_how_do_i(matcher)

    run_patterns_dev_set(matcher)
    #run_patterns_test_set(matcher)
