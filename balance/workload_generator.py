import os
import copy
import logging
import random
from typing import List
import pickle

import numpy as np

import balance.embedding_utils as embedding_utils
from index_selection_evaluation.selection.candidate_generation import (
    candidates_per_query,
    syntactically_relevant_indexes,
)
from index_selection_evaluation.selection.cost_evaluation import CostEvaluation
from index_selection_evaluation.selection.dbms.postgres_dbms import PostgresDatabaseConnector
from index_selection_evaluation.selection.utils import get_utilized_indexes
from index_selection_evaluation.selection.workload import Query, Workload

from .workload_embedder import WorkloadEmbedder

QUERY_PATH = "query_files"

def get_query_texts_from_file(file: str) -> List[str]:
    with open(file, 'r') as f:
        sqls = f.readlines()
    if sqls[-1] == '':
        sqls.pop()
    return sqls


class WorkloadGenerator(object):
    def __init__(
        self, wk_config, workload_columns, random_seed, database_name, experiment_id=None,
        filter_utilized_columns=None,experiment_folder_path=None,spath=None,
        tpl2tid={}, dummy=False,
        input_workload=None, input_worklod_path='',
        weight_path='',shuffle=False
    ):
        self.benchmark = wk_config["benchmark"]
        assert self.benchmark  in [
            "TPCH",
            "TPCDS",
            "JOB",
            "TPCHC",
            "TPCDSC",
            "CEB"
        ], f"Benchmark '{self.benchmark}' is currently not supported."

        # For create view statement differentiation
        self.experiment_id = experiment_id
        self.filter_utilized_columns = filter_utilized_columns
        self.path2 = spath
        self.gen_one = 0
        self.temp_genone = []

        self.rnd = random.Random()
        self.rnd.seed(random_seed)
        self.np_rnd = np.random.default_rng(seed=random_seed)

        self.workload_columns = workload_columns
        self.database_name = database_name
        self.tpl2tid = tpl2tid

        self.number_of_query_classes = self._set_number_of_query_classes()  # set wrt benchmark
        self.excluded_query_classes = set(wk_config["excluded_query_classes"])
        if dummy:
            self.varying_frequencies = False
        else:
            self.varying_frequencies = wk_config["varying_frequencies"]
        validation_instances = wk_config["validation_testing"]["number_of_workloads"]
        test_instances = wk_config["validation_testing"]["number_of_workloads"]

        # self.query_texts is list of lists. Outer list for query classes, inner list for instances of this class.
        #self.query_texts = self._retrieve_query_texts()
        self.query_texts: List[List[str]] = []
        if not input_workload and not input_worklod_path:
            self.query_texts = self._retrieve_query_texts_random_value()  # get queries texts from folders, e.g., TPCDSC/TPCDSC_*.txt
        elif input_worklod_path:
            self.query_texts = [[qstr] for qstr in get_query_texts_from_file(input_worklod_path)]
        else:
            self.query_texts = [[q.text] for q in input_workload.queries]
        self.query_classes = set(range(1, self.number_of_query_classes + 1))
        self.available_query_classes = self.query_classes - self.excluded_query_classes

        self.globally_indexable_columns = self._select_indexable_columns(self.filter_utilized_columns) # obtain indexable columns wrt self.query_texts

        self.wl_validation = []
        self.wl_testing = []

        # __import__('pdb').set_trace()
        if False:
        # if wk_config["similar_workloads"] and wk_config["unknown_queries"] == 0:  # similar workloads with all known queries
            # Todo: this branch can probably be removed
            assert self.varying_frequencies, "Similar workloads can only be created with varying frequencies."
            self.wl_validation = [None]
            self.wl_testing = [None]
            _, self.wl_validation[0], self.wl_testing[0] = self._generate_workloads(
                0, validation_instances, test_instances, wk_config["size"], weight_path=weight_path, shuffle=shuffle
            )
            if wk_config["query_class_change_frequency"] is None:
                self.wl_training = self._generate_similar_workloads(wk_config["training_instances"], wk_config["size"])
            else:
                self.wl_training = self._generate_similar_workloads_qccf(
                    wk_config["training_instances"], wk_config["size"], wk_config["query_class_change_frequency"]
                )
        elif False:
        # elif wk_config["unknown_queries"] > 0 and wk_config["validation_testing"]["unknown_query_probabilities"][-1] > 0.01:
            # with unknown queries
            embedder_connector = PostgresDatabaseConnector(self.database_name, autocommit=True)
            embedder = WorkloadEmbedder(
                # Transform globally_indexable_columns to list of lists.
                self.query_texts,
                0,
                embedder_connector,
                [list(map(lambda x: [x], self.globally_indexable_columns))],
                retrieve_plans=True,
            )
            self.unknown_query_classes = embedding_utils.which_queries_to_remove(
                embedder.plans, wk_config["unknown_queries"], random_seed,experiment_folder_path=experiment_folder_path
            )

            self.unknown_query_classes = frozenset(self.unknown_query_classes) - self.excluded_query_classes
            missing_classes = wk_config["unknown_queries"] - len(self.unknown_query_classes)
            self.unknown_query_classes = self.unknown_query_classes | frozenset(
                self.rnd.sample(self.available_query_classes - frozenset(self.unknown_query_classes), missing_classes)
            )
            assert len(self.unknown_query_classes) == wk_config["unknown_queries"]

            self.known_query_classes = self.available_query_classes - frozenset(self.unknown_query_classes)
            embedder = None

            for query_class in self.excluded_query_classes:
                assert query_class not in self.unknown_query_classes

            logging.critical(f"Global unknown query classes: {sorted(self.unknown_query_classes)}")
            logging.critical(f"Global known query classes: {sorted(self.known_query_classes)}")

            for unknown_query_probability in wk_config["validation_testing"]["unknown_query_probabilities"]:
                _, wl_validation, wl_testing = self._generate_workloads(
                    0,
                    validation_instances,
                    test_instances,
                    wk_config["size"],
                    unknown_query_probability=unknown_query_probability,
                    weight_path=weight_path, shuffle=shuffle
                )
                self.wl_validation.append(wl_validation)
                self.wl_testing.append(wl_testing)

            assert (
                len(self.wl_validation)
                == len(wk_config["validation_testing"]["unknown_query_probabilities"])
                == len(self.wl_testing)
            ), "Validation/Testing workloads length fail"

            # We are temporarily restricting the available query classes now to exclude certain classes for training
            original_available_query_classes = self.available_query_classes
            self.available_query_classes = self.known_query_classes

            # generate self.wl_training, similar & cf OR similar OR train+validation+test
            if wk_config["similar_workloads"]:
                if wk_config["query_class_change_frequency"] is not None:
                    logging.critical(
                        f"Similar workloads with query_class_change_frequency: {wk_config['query_class_change_frequency']}"
                    )
                    self.wl_training = self._generate_similar_workloads_qccf(
                        wk_config["training_instances"], wk_config["size"], wk_config["query_class_change_frequency"]
                    )
                else:
                    self.wl_training = self._generate_similar_workloads(wk_config["training_instances"], wk_config["size"])
            else:
                self.wl_training, _, _ = self._generate_workloads(wk_config["training_instances"], 0, 0, wk_config["size"], weight_path=weight_path, shuffle=shuffle)
            # We are removing the restriction now.
            self.available_query_classes = original_available_query_classes
        elif False:
        # elif wk_config["unknown_queries"] > 0 and wk_config["validation_testing"]["unknown_query_probabilities"][-1] <= 0.01:
            # assert (
            #     config["validation_testing"]["unknown_query_probabilities"][-1] > 0
            # ), "Query unknown_query_probabilities should be larger 0."

            embedder_connector = PostgresDatabaseConnector(self.database_name, autocommit=True)
            embedder = WorkloadEmbedder(
                # Transform globally_indexable_columns to list of lists.
                self.query_texts,
                0,
                embedder_connector,
                [list(map(lambda x: [x], self.globally_indexable_columns))],
                retrieve_plans=True,
            )
            self.unknown_query_classes = embedding_utils.which_queries_to_remove(
                embedder.plans, wk_config["unknown_queries"], random_seed,experiment_id,experiment_folder_path=experiment_folder_path
            )

            self.unknown_query_classes = frozenset(self.unknown_query_classes) - self.excluded_query_classes
            missing_classes = wk_config["unknown_queries"] - len(self.unknown_query_classes)
            self.unknown_query_classes = self.unknown_query_classes | frozenset(
                self.rnd.sample(self.available_query_classes - frozenset(self.unknown_query_classes), missing_classes)
            )
            assert len(self.unknown_query_classes) == wk_config["unknown_queries"]

            self.known_query_classes = self.available_query_classes - frozenset(self.unknown_query_classes)
            embedder = None

            for query_class in self.excluded_query_classes:
                assert query_class not in self.unknown_query_classes

            logging.critical(f"Global unknown query classes: {sorted(self.unknown_query_classes)}")
            logging.critical(f"Global known query classes: {sorted(self.known_query_classes)}")


            _, wl_validation, wl_testing = self._generate_workloads(
                    0,
                    validation_instances,
                    test_instances,
                    wk_config["size"]
            )
            self.wl_validation.append(wl_validation)
            self.wl_testing.append(wl_testing)

            assert (
                len(self.wl_validation)
                == len(wk_config["validation_testing"]["unknown_query_probabilities"])
                == len(self.wl_testing)
            ), "Validation/Testing workloads length fail"

            # We are temporarily restricting the available query classes now to exclude certain classes for training
            original_available_query_classes = self.available_query_classes
            self.available_query_classes = self.known_query_classes

            if wk_config["similar_workloads"]:
                if wk_config["query_class_change_frequency"] is not None:
                    logging.critical(
                        f"Similar workloads with query_class_change_frequency: {wk_config['query_class_change_frequency']}"
                    )
                    self.wl_training = self._generate_similar_workloads_qccf(
                        wk_config["training_instances"], wk_config["size"], wk_config["query_class_change_frequency"]
                    )
                else:
                    self.wl_training = self._generate_similar_workloads(wk_config["training_instances"], wk_config["size"])
            else:
                self.wl_training, _, _ = self._generate_workloads(wk_config["training_instances"], 0, 0, wk_config["size"], weight_path=weight_path, shuffle=shuffle)
            # We are removing the restriction now.
            self.available_query_classes = original_available_query_classes
        # else:

        if True:
            self.wl_validation = [None]
            self.wl_testing = [None]
            if not input_workload and not input_worklod_path:  # generated workloads with random orders with _generate_random_workload()
                self.wl_training, self.wl_validation[0], self.wl_testing[0] = self._generate_workloads(
                            wk_config["training_instances"], validation_instances, test_instances, wk_config["size"],
                            weight_path=weight_path, shuffle=shuffle
                    )
            else:
                workload_class_order, workload_class_freq = self._generate_random_workload(wk_config["size"], weight_path=weight_path, shuffle=shuffle)
                if input_workload:
                    pass
                else:  # input_worklod_path
                    assert input_worklod_path
                    queries = []
                    for i in range(wk_config['size']):
                        q = Query(query_id=i+1,query_text= self.query_texts[i][0], frequency=workload_class_freq[i])
                        self._store_indexable_columns(q)
                        queries.append(q)
                    input_workload = Workload(queries)
                input_workload.queries =  [input_workload.queries[workload_class_order[i]-1] for i in  range(wk_config['size'])]

                self.wl_training = [input_workload]
                self.wl_validation = [[input_workload]]
                self.wl_testing = [[input_workload]]

        # logging.critical(f"Sample training workloads: {self.rnd.sample(self.wl_training, 1)}")
        logging.info("Finished generating workloads.")



    def _retrieve_query_texts_random_value(self):
        query_files = [
            open(f"{QUERY_PATH}/{self.benchmark}/{self.benchmark}_{file_number}.txt", "r")
            for file_number in range(1, self.number_of_query_classes + 1)
        ]

        finished_queries: List[str] = []
        for query_file in query_files:

            queries = query_file.readlines()
            qq = []
            for i in range(len(queries)):
                now_q = queries[i:i+1]
                now_q = self._preprocess_queries(now_q)
                qq.append(now_q)
            finished_queries.append(qq)

            query_file.close()

        assert len(finished_queries) == self.number_of_query_classes

        return finished_queries

    def _set_number_of_query_classes(self):
        if self.benchmark == "TPCH":
            return 22
        elif self.benchmark == "TPCHC":
            return 20
        elif self.benchmark == "TPCDS":
            return 99
        elif self.benchmark == "TPCDSC":
            return 20
        elif self.benchmark == "JOB":
            return 113  # return 113
        elif self.benchmark == "CEB":
            return 16  # return 113
        else:
            raise ValueError("Unsupported Benchmark type provided, only TPCH, TPCDS, and JOB supported.")

    def _retrieve_query_texts(self):
        query_files = [
            open(f"{QUERY_PATH}/{self.benchmark}/{self.benchmark}_{file_number}.txt", "r")
            for file_number in range(1, self.number_of_query_classes + 1)
        ]

        finished_queries = []
        for query_file in query_files:
            queries = query_file.readlines()[:1]
            queries = self._preprocess_queries(queries)

            finished_queries.append(queries)

            query_file.close()

        assert len(finished_queries) == self.number_of_query_classes

        return finished_queries

    def _preprocess_queries(self, queries) -> List[str]:
        """
        remove limit clauses?
        and create view with exp unique name
        """
        processed_queries = []
        for query in queries:
            query = query.replace("limit 100", "")
            query = query.replace("limit 20", "")
            query = query.replace("limit 10", "")
            query = query.strip()

            if "create view revenue0" in query:
                query = query.replace("revenue0", f"revenue0_{self.experiment_id}")

            processed_queries.append(query)

        return processed_queries

    def _store_indexable_columns(self, query):
        """
        referenced columns for non-JOB benchmark;
        for JOB, only accounts for the columns after keyword WHERE
        """
        if self.benchmark != "JOB":
            # TODO: select filtered db columns where their names in the query text,
            # there're issues with this rule, but it's fine with 22 query class in tpc-h
            # e.g., part and partsupp always co-occurs.
            for column in self.workload_columns:
                if column.name in query.text:
                    query.columns.append(column)
        else:
            query_text = query.text
            assert "WHERE" in query_text, f"Query without WHERE clause encountered: {query_text} in {query.nr}"

            split = query_text.split("WHERE")
            assert len(split) == 2, "Query split for JOB query contains subquery"
            query_text_before_where = split[0]
            query_text_after_where = split[1]

            for column in self.workload_columns:
                if column.name in query_text_after_where and f"{column.table.name} " in query_text_before_where:
                    query.columns.append(column)

    def _workloads_from_tuples(self, tuples, unknown_query_probability=None) -> List[Workload]:
        """
        generate workloads according to selected templates in *tuples* wrt self.query_texts
        """
        workloads = []
        unknown_query_probability = "" if unknown_query_probability is None else unknown_query_probability

        for tupl in tuples:
            query_classes, query_class_frequencies = tupl
            queries = []

            for query_class, frequency in zip(query_classes, query_class_frequencies):
                query_text = self.rnd.choice(self.query_texts[query_class - 1])
                if not isinstance(query_text,  str) and isinstance(query_text, list):
                    query_text = query_text[0]

                query = Query(query_class, query_text, frequency=frequency)

                self._store_indexable_columns(query)
                assert len(query.columns) > 0, f"Query columns should have length > 0: {query.text}"

                queries.append(query)

            assert isinstance(queries, list), f"Queries is not of type list but of {type(queries)}"
            previously_unseen_queries = (
                round(unknown_query_probability * len(queries)) if unknown_query_probability != "" else 0
            )
            workloads.append(
                Workload(queries, description=f"Contains {previously_unseen_queries} previously unseen queries.")
            )

        return workloads

    def _generate_workloads(
        self, train_instances, validation_instances, test_instances, size, unknown_query_probability=None, weight_path='', shuffle=False
    ):
        required_unique_workloads = train_instances + validation_instances + test_instances

        unique_workload_tuples = set()
        # sample *required_unique_workloads* number of workloads
        while required_unique_workloads > len(unique_workload_tuples):
            workload_tuple = self._generate_random_workload(size, unknown_query_probability, weight_path, shuffle)
            unique_workload_tuples.add(workload_tuple)
            # if not self.varying_frequencies and not shuffle:
            if True:
                validation_instances = 1
                test_instances = 1
                break

        validation_tuples = self.rnd.sample(unique_workload_tuples, validation_instances)
        test_workload_tuples = self.rnd.sample(unique_workload_tuples, test_instances)

        if self.varying_frequencies and shuffle:
            unique_workload_tuples = unique_workload_tuples - set(validation_tuples)
            unique_workload_tuples = unique_workload_tuples - set(test_workload_tuples)

        train_workload_tuples = unique_workload_tuples

        if self.varying_frequencies and shuffle:
            assert (
                len(train_workload_tuples) + len(test_workload_tuples) + len(validation_tuples) == required_unique_workloads
            )



        if True:
        # if not os.path.exists(self.path2):
            validation_workloads = self._workloads_from_tuples(validation_tuples, unknown_query_probability)
            test_workloads = self._workloads_from_tuples(test_workload_tuples, unknown_query_probability)
            train_workloads = self._workloads_from_tuples(train_workload_tuples, unknown_query_probability)
            return train_workloads, validation_workloads, test_workloads
        else:
            import joblib
            pp = self.path2
            with open(pp, 'rb') as f:
                wl1 = joblib.load(f)  # 10000 samples with same templated queries with different frequencies?
                f.close()


            latest_200_samples = wl1[-200:]
            latest_250_to_200_samples = wl1[-250:-200]

            train_workloads=[]
            test_workloads=[]
            validation_workloads=[]


            while len(train_workloads)<train_instances:
                train_workloads.extend(self.rnd.sample(latest_200_samples, 50))

            while len(test_workloads)<validation_instances:
                test_workloads.extend(self.rnd.sample(latest_250_to_200_samples, 10))

            while len(validation_workloads)<test_instances:
                validation_workloads.extend(self.rnd.sample(latest_250_to_200_samples, 10))
            return train_workloads, test_workloads, validation_workloads


    # The core idea is to create workloads that are similar and only change slightly from one to another.
    # For the following workload, we remove one random element, add another random one with frequency, and
    # randomly change the frequency of one element (including the new one).
    def _generate_similar_workloads(self, instances, size):
        # remove & insert a query instance that is drawn from random query classes
        # for a *size* of times
        assert size <= len(
            self.available_query_classes
        ), "Cannot generate workload with more queries than query classes"

        workload_tuples = []

        query_classes = self.rnd.sample(self.available_query_classes, size)
        available_query_classes = self.available_query_classes - frozenset(query_classes)
        frequencies = list(self.np_rnd.zipf(1.5, size))

        workload_tuples.append((copy.copy(query_classes), copy.copy(frequencies)))

        for workload_idx in range(instances - 1):
            # Remove a random element
            idx_to_remove = self.rnd.randrange(len(query_classes))
            query_classes.pop(idx_to_remove)
            frequencies.pop(idx_to_remove)

            # Draw a new random element, the removed one is excluded
            query_classes.append(self.rnd.sample(available_query_classes, 1)[0])
            frequencies.append(self.np_rnd.zipf(1.5, 1)[0])

            frequencies[self.rnd.randrange(len(query_classes))] = self.np_rnd.zipf(1.5, 1)[0]

            available_query_classes = self.available_query_classes - frozenset(query_classes)
            workload_tuples.append((copy.copy(query_classes), copy.copy(frequencies)))

        workloads = self._workloads_from_tuples(workload_tuples)

        return workloads

    # This version uses the same query id selction for query_class_change_frequency workloads
    def _generate_similar_workloads_qccf(self, instances, size, query_class_change_frequency):
        # random sample a *size* of query class along with a random freq
        assert size <= len(
            self.available_query_classes
        ), "Cannot generate workload with more queries than query classes"

        workload_tuples = []

        while len(workload_tuples) < instances:
            if len(workload_tuples) % query_class_change_frequency == 0:
                query_classes = self.rnd.sample(self.available_query_classes, size)

            frequencies = list(self.np_rnd.integers(1, 10000, size))
            workload_tuples.append((copy.copy(query_classes), copy.copy(frequencies)))

        workloads = self._workloads_from_tuples(workload_tuples)

        return workloads

    def _generate_random_workload(self, size, unknown_query_probability=None, weight_path='', shuffle=False):
        assert size <= self.number_of_query_classes, "Cannot generate workload with more queries than query classes"

        workload_query_classes = None
        if unknown_query_probability is not None:
            number_of_unknown_queries = round(size * unknown_query_probability)
            number_of_known_queries = size - number_of_unknown_queries
            assert number_of_known_queries + number_of_unknown_queries == size

            known_query_classes = self.rnd.sample(self.known_query_classes, number_of_known_queries)
            unknown_query_classes = self.rnd.sample(self.unknown_query_classes, number_of_unknown_queries)
            query_classes = known_query_classes
            query_classes.extend(unknown_query_classes)
            workload_query_classes = tuple(query_classes)
            assert len(workload_query_classes) == size
        else:
            if len(self.available_query_classes)<size:
                size = len(self.available_query_classes)
            if False: # if self.gen_one>0+9:
                workload_query_classes = self.rnd.choice(self.temp_genone)
            else:
                workload_query_classes = tuple(self.rnd.sample(self.available_query_classes, size))
                self.temp_genone.append(workload_query_classes)
            self.gen_one = self.gen_one +1

        # Create frequencies
        if self.varying_frequencies:
            if not weight_path:
                query_class_frequencies = tuple(list(self.np_rnd.integers(1, 10000, size)))
            else:
                with open(weight_path, 'rb') as f:
                    weight_list = pickle.load(f)
                weight_list2 = [weight_list[i-1] for i in workload_query_classes]
                query_class_frequencies = tuple(weight_list2)
        else:
            query_class_frequencies = tuple([1 for frequency in range(size)])

        workload_tuple = (workload_query_classes, query_class_frequencies)

        return workload_tuple

    def _only_utilized_indexes(self, indexable_columns):
        frequencies = [1 for frequency in range(len(self.available_query_classes))]
        workload_tuple = (self.available_query_classes, frequencies)
        workload = self._workloads_from_tuples([workload_tuple])[0]

        candidates = candidates_per_query(
            workload,
            max_index_width=1,
            candidate_generator=syntactically_relevant_indexes,
        )

        connector = PostgresDatabaseConnector(self.database_name, autocommit=True)
        connector.drop_indexes()
        cost_evaluation = CostEvaluation(connector)

        utilized_indexes, query_details = get_utilized_indexes(workload, candidates, cost_evaluation, True)

        columns_of_utilized_indexes = set()
        for utilized_index in utilized_indexes:
            column = utilized_index.columns[0]
            columns_of_utilized_indexes.add(column)

        output_columns = columns_of_utilized_indexes & set(indexable_columns)
        excluded_columns = set(indexable_columns) - output_columns
        logging.critical(f"Excluding columns based on utilization:\n   {excluded_columns}")

        return output_columns

    def _select_indexable_columns(self, only_utilized_indexes=False):
        available_query_classes = tuple(self.available_query_classes)
        query_class_frequencies = tuple([1 for frequency in range(len(available_query_classes))])

        logging.info(f"Selecting indexable columns on {len(available_query_classes)} query classes.")

        workload = self._workloads_from_tuples([(available_query_classes, query_class_frequencies)])[0]

        indexable_columns = workload.indexable_columns()
        # only_utilized_indexes = True  # debug line, for DB2Advis like optimizer chosen indexes
        if only_utilized_indexes:
            indexable_columns = self._only_utilized_indexes(indexable_columns)
        selected_columns = []

        global_column_id = 0
        for column in self.workload_columns:
            if column in indexable_columns:
                column.global_column_id = global_column_id
                global_column_id += 1

                selected_columns.append(column)

        return selected_columns
