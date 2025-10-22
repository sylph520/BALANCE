import importlib
import logging

from index_selection_evaluation.selection.dbms.postgres_dbms import PostgresDatabaseConnector
from index_selection_evaluation.selection.table_generator import TableGenerator


class Schema(object):
    def __init__(self, benchmark_name, scale_factor, dbnames, filters={}):
        generating_connector = PostgresDatabaseConnector(None, autocommit=True)

        # obtain table_generator, create database if not exists and obtains instances of tables and columns (from ddl stmt and db conncetor)
        table_generator = TableGenerator(
            benchmark_name=benchmark_name.lower(), scale_factor=scale_factor, database_connector=generating_connector,dbname=dbnames
        )

        self.database_name = table_generator.database_name()
        self.tables = table_generator.tables
        self.types = table_generator.types

        self.columns = []
        for table in self.tables:
            for column in table.columns:
                self.columns.append(column)

        if 'tpchc' not in benchmark_name.lower() or 'tpcdsc' not in benchmark_name.lower():
            for filter_name in filters.keys():  # e.g., TableNumRowsFilter
                filter_class = getattr(importlib.import_module("balance.schema"), filter_name)
                filter_instance = filter_class(filters[filter_name], self.database_name)
                self.columns = filter_instance.apply_filter(self.columns)


class TableNumRowsFilter(object):
    def __init__(self, threshold, database_name):
        self.threshold = threshold
        self.connector = PostgresDatabaseConnector(database_name, autocommit=True)
        self.connector.create_statistics()

    def apply_filter(self, columns):
        output_columns = []

        for column in columns:
            table_name = column.table.name
            table_num_rows = self.connector.exec_fetch(
                f"SELECT reltuples::bigint AS estimate FROM pg_class where relname='{table_name}';"
            )[0]

            if table_num_rows > self.threshold:
                output_columns.append(column)

        logging.warning(f"Reduced columns from {len(columns)} to {len(output_columns)}.")

        return output_columns
