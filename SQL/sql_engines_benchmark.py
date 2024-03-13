import sqlite3 as sql
import pymonetdb
from tableauhyperapi import HyperProcess, Telemetry, CreateMode, Connection
from timeit import default_timer as timer

def time_hyper_query(query, parameters):
    mode = "compiled" if parameters["initial_compilation_mode"] == "o" else "interpreted"

    with HyperProcess(telemetry=Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU, parameters=parameters) as hyper:
        with Connection(hyper.endpoint, "SQL/data.hyper", CreateMode.CREATE_AND_REPLACE) as connection:
            # plan
            tic = timer()
            hyper_res = connection.execute_list_query("EXPLAIN " + query)
            toc = timer()

            time_planning = toc - tic

            # execute
            tic = timer()
            hyper_res = connection.execute_list_query(query)
            toc = timer()
    
    print(f"hyper ({mode}) result: {hyper_res}\n(computed in {toc - tic - time_planning}s) - planning time: {time_planning}")


def time_sqlite_query(query):
    db_connection = sql.connect("SQL/test.db")
    db = db_connection.cursor()
    res = db.execute(query)

    # plan
    tic = timer()
    sql_res = db.execute("EXPLAIN QUERY PLAN " + query)
    sql_res = sql_res.fetchall()
    toc = timer()

    time_planning = toc - tic

    # execute
    tic = timer()
    sql_res = db.execute(query)
    sql_res = sql_res.fetchall()
    toc = timer()
    
    print(f"sqlite result: {sql_res}\n(computed in {toc - tic - time_planning}s) - planning time: {time_planning}")



if __name__ == "__main__":
    sql_file = "SQL/test_query.sql"

    with open(sql_file, "r") as file:
        query = file.read()


    # ----- HYPER ------
        
    # hyper - compiled
    parameters = {
        "log_config": "",
        "max_query_size": "1000000",
        "hard_concurrent_query_thread_limit": "1",
        "initial_compilation_mode": "o"
    }

    time_hyper_query(query=query, parameters=parameters)

    # hyper - interpreted
    parameters = {
        "log_config": "",
        "max_query_size": "1000000",
        "hard_concurrent_query_thread_limit": "1",
        "initial_compilation_mode": "v"
    }

    time_hyper_query(query=query, parameters=parameters)


    # ------ SQLite ------

    time_sqlite_query(query)


    # ------ MonetDB ------

    # connection = pymonetdb.connect(username="monetdb", password="monetdb", hostname="localhost", database="demo")
    # cursor = connection.cursor()
    # cursor.execute(query)
    # res = cursor.fetchall()

    # print(res)