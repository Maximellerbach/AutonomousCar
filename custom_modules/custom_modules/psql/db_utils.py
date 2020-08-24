from configparser import ConfigParser

import psycopg2


def _config(filename='database.ini', section='postgresql'):
    parser = ConfigParser()
    parser.read(filename)

    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception(
            'Section {0} not found in the {1} file'.format(section, filename))

    return db


def connect():
    return psycopg2.connect(**_config())


def _start_db(filename='database.ini', section='database'):
    import subprocess
    parser = ConfigParser()
    parser.read(filename)

    if parser.has_section(section):
        items = parser.items(section)
        for item in items:
            db_path = item[1]

            command = f"pg_ctl start -D {db_path}"
            subprocess.run(command)


def start_if_not_running(filename='database.ini', section='database'):
    try:
        conn = connect()
    except psycopg2.Error:
        _start_db(filename=filename, section=section)
