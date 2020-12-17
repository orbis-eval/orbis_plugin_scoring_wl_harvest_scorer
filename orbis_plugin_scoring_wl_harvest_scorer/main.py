# -*- coding: utf-8 -*-
"""Summary

Attributes:
    logger (TYPE): Description
"""

import logging
from datetime import datetime
from operator import itemgetter

import Levenshtein
import editdistance
from fuzzywuzzy import fuzz
from nilsimsa import Nilsimsa
from nilsimsa import compare_digests
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer

from orbis_eval.core.base import PluginBaseClass

logger = logging.getLogger(__name__)


class Main(PluginBaseClass):
    """Summary
    """

    def __init__(self):
        """Summary
        """
        super(Main, self).__init__()

    def run(self, computed, gold, scorer_condition):
        """Summary

        Args:
            computed (TYPE): Description
            gold (TYPE): Description
            scorer_condition (TYPE): Description

        Returns:
            TYPE: Description
        """

        gold_0 = gold.copy()
        computed_0 = computed.copy()
        entity_mappings = []

        with open("harvest_evaluation_results.txt", "w") as open_file:
            open_file.write("")

        msg = f"Harvest Evaluation Results ({datetime.now()})"
        entity_mappings, gold_0, computed_0, msg = self.get_scored(
            entity_mappings, gold_0, computed_0, scorer_condition, msg)
        entity_mappings, computed_0, msg = self.get_unscored(
            entity_mappings, computed_0, msg)
        confusion_matrix = self.get_confusion_matrix(entity_mappings)

        with open("harvest_evaluation_results.txt", "+a") as open_file:
            open_file.write(msg)

        print(msg)

        return confusion_matrix

    def get_scored(self, entity_mappings, gold_0, computed_0, scorer_condition, msg):
        """Summary

        Args:
            entity_mappings (TYPE): Description
            gold_0 (TYPE): Description
            computed_0 (TYPE): Description
            scorer_condition (TYPE): Description

        Returns:
            TYPE: Description
        """

        for gold_entry in sorted(gold_0, key=itemgetter("start")):
            gold_start = int(gold_entry["start"])
            gold_end = int(gold_entry["end"])
            gold_id = "{},{}".format(gold_start, gold_end)
            gold_url = gold_entry["key"]
            gold_type = gold_entry["entity_type"].lower()
            gold_surface_form = gold_entry["surfaceForm"]
            entity_mapping = [gold_id, False, 0, "fn"]

            for comp_entry in sorted(computed_0, key=itemgetter("document_start")):
                comp_start = int(comp_entry["document_start"])
                comp_end = int(comp_entry["document_end"])
                comp_url = comp_entry["key"]
                comp_type = comp_entry["entity_type"].lower()
                comp_surface_form = comp_entry["surfaceForm"]

                score = self.calc_fuzzyration(gold_surface_form, comp_surface_form)
                # score = self.calc_jaccard(gold_surface_form, comp_surface_form)
                # score = self.calc_levenshtein(gold_surface_form, comp_surface_form)
                # score = self.calc_editdistance(gold_surface_form, comp_surface_form)
                # score = self.calc_nilsimsa(gold_surface_form, comp_surface_form)

                states = {
                    "same_url": gold_url == comp_url,
                    "same_type": gold_type == comp_type,
                    "same_surface_form": gold_surface_form == comp_surface_form,
                    "same_start": gold_start == comp_start,
                    "same_end": gold_end == comp_end,
                    "overlap": gold_end >= comp_start and gold_start <= comp_end,
                    "similarity": score
                }

                best_condition = all([
                    states['same_url'],
                    states['same_type'],
                    states['same_surface_form'],
                    states['same_start'],
                    states['same_end']
                ])

                min_condition = all([
                    states['same_url'],
                    states['same_type'],
                    states['overlap']
                ])

                if best_condition or min_condition:
                    msg += "\nScored (TP)\n"
                    msg += f"{42 * '#'}\n"
                    msg += "\n"
                    msg += f"gold surface: \n{gold_surface_form}\n"
                    msg += f"comp surface: \n{comp_surface_form}\n"
                    msg += f">Similarity: {states['similarity']}\n"
                    msg += "\n"
                    msg += f"gold_url:   {gold_url}\n"
                    msg += f"comp_url:   {comp_url}\n"
                    msg += f">same_url:  {gold_url == comp_url}\n"
                    msg += "\n"
                    msg += f"gold_type:  {gold_type}\n"
                    msg += f"comp_type:  {comp_type}\n"
                    msg += f">same_type: {gold_type == comp_type}\n"
                    msg += "\n"
                    msg += f"gold_start: {gold_start}\n"
                    msg += f"comp_start: {comp_start}\n"
                    msg += f">same_start: {gold_start == comp_start}\n"
                    msg += "\n"
                    msg += f"gold_end:   {gold_end}\n"
                    msg += f"comp_end:   {comp_end}\n"
                    msg += f">same_end:   {gold_end == comp_end}\n"

                    msg += f"\nSimilarity Results:\n"
                    # msg += f"Levenshtein: {levenshtein_score}\n"
                    # msg += f"Edit Distance: {editdistance_score}\n"
                    msg += f"fuzzywuzzy: {score}\n"
                    # msg += f"jaccard: {jaccard_score}\n"
                    # msg += f"Nilsimsa: {nilsimsa_score}\n"
                    msg += "\n"

                # multiline_logging(app, states)
                if best_condition:
                    # entity_mapping = [gold_id, False, 0, "fn"]

                    msg += "Score: +1 (all conditions met)\n"
                    comp_id = f"{comp_start},{comp_end}"
                    entity_mapping[1] = comp_id
                    entity_mapping[2] += 1
                    entity_mapping[3] = states
                    gold_0.remove(gold_entry)
                    computed_0.remove(comp_entry)
                    break
                elif min_condition:
                    # score = self.calc_score(states)
                    # score = (states['similarity'] / 100)
                    score = (states['similarity'])
                    msg += f"Score: +{score} (FuzzyWuzzy similarity based)\n"
                    comp_id = f"{comp_start},{comp_end}"
                    entity_mapping[1] = comp_id
                    entity_mapping[2] += score
                    entity_mapping[3] = states
                    gold_0.remove(gold_entry)
                    computed_0.remove(comp_entry)
                    break
                else:
                    continue

            entity_mappings.append(entity_mapping)

        return entity_mappings, gold_0, computed_0, msg

    def calc_nilsimsa(self, gold_surface_form, comp_surface_form):
        nil_0 = Nilsimsa(gold_surface_form)
        nil_1 = Nilsimsa(comp_surface_form)
        nil = compare_digests(nil_0.hexdigest(), nil_1.hexdigest())
        return nil

    def calc_editdistance(self, gold_surface_form, comp_surface_form):
        return editdistance.eval(gold_surface_form, comp_surface_form)

    def calc_levenshtein(self, gold_surface_form, comp_surface_form):
        return Levenshtein.distance(gold_surface_form, comp_surface_form)

    def calc_fuzzyration(self, gold_surface_form, comp_surface_form):
        return fuzz.ratio(gold_surface_form, comp_surface_form) / 100

    def calc_jaccard(self, gold_surface_form, comp_surface_form):
        try:
            vectorizer = CountVectorizer(lowercase=False)
            vectorizer.fit([gold_surface_form, comp_surface_form])

            gold_vector = vectorizer.transform([gold_surface_form]).toarray().tolist()[0]
            comp_vector = vectorizer.transform([comp_surface_form]).toarray().tolist()[0]

            # jaccard = metrics.jaccard_score(gold_vector, comp_vector, average=None)
            jaccard = metrics.jaccard_similarity_score(gold_vector, comp_vector)
        except ValueError:
            logger.warning(f"Could not calculate jaccard for gold [{gold_surface_form}] computed [{comp_surface_form}]")
            jaccard = 0
        return jaccard

    def calc_score(self, states):
        """
        noting yet...

        Args:
            states (TYPE): Description

        Returns:
            TYPE: Description
        """
        right, wrong = 0, 0

        for k, v in states.items():
            if v:
                right += 1
            else:
                wrong += 1

        return 0

    def get_unscored(self, entity_mappings, computed_0, msg):
        """Summary

        Args:
            entity_mappings (TYPE): Description
            computed_0 (TYPE): Description

        Returns:
            TYPE: Description
        """

        for comp_entry in sorted(computed_0, key=itemgetter("document_start")):
            comp_start = int(comp_entry["document_start"])
            comp_end = int(comp_entry["document_end"])
            comp_url = comp_entry["key"]
            comp_doc_id = comp_entry["key"]
            comp_type = comp_entry["entity_type"].lower()
            comp_surface_form = comp_entry["surfaceForm"]
            comp_id = f"{comp_start},{comp_end}"
            entity_mapping = [False, comp_id, 0, "fp"]
            entity_mappings.append(entity_mapping)

            """
            msg += "\n\nUnscored (FP)\n"
            msg += f"{42 * '#'}\n"
            msg += f"comp surface: \n{comp_surface_form}\n"
            msg += f"comp_url:   {comp_url}\n"
            msg += f"comp_type:  {comp_type}\n"
            msg += f"comp_start: {comp_start}\n"
            msg += f"comp_end:   {comp_end}\n"
            # """

        return entity_mappings, computed_0, msg

    def get_confusion_matrix(self, entity_mappings):
        """Summary

        Args:
            entity_mappings (TYPE): Description

        Returns:
            TYPE: Description

        Raises:
            RuntimeError: Description
        """

        confusion_matrix = {
            "tp": [],
            "fp": [],
            "fn": [],
            "tp_ids": [],
            "fp_ids": [],
            "fn_ids": [],
            "states": []
        }

        for gold, comp, num, state in entity_mappings:

            confusion_matrix["states"].append(state)

            if gold and comp:
                # logger.debug("TP: Gold: {}; Comp: {}; ({})".format(gold, comp, num))
                confusion_matrix["tp"].append(num)
                confusion_matrix["fp"].append(1 - num)
                confusion_matrix["fn"].append(0)
                confusion_matrix["tp_ids"].append(comp)

            elif comp and not gold:
                # logger.debug("FP: Gold: {}; Comp: {}; ({})".format(gold, comp, num))
                confusion_matrix["tp"].append(0)
                confusion_matrix["fp"].append(1)
                confusion_matrix["fn"].append(0)
                confusion_matrix["fp_ids"].append(comp)

            elif gold and not comp:
                # logger.debug("FN: Gold: {}; Comp: {}; ({})".format(gold, comp, num))
                confusion_matrix["tp"].append(0)
                confusion_matrix["fp"].append(0)
                confusion_matrix["fn"].append(1)
                confusion_matrix["fn_ids"].append(gold)

            elif not gold and not comp:
                raise RuntimeError

            else:
                # print("Error")
                pass

        confusion_matrix["tp_sum"] = sum(confusion_matrix["tp"])
        confusion_matrix["fp_sum"] = sum(confusion_matrix["fp"])
        confusion_matrix["fn_sum"] = sum(confusion_matrix["fn"])

        return confusion_matrix
