from custom_modules.psql.queries import label
from custom_modules.psql import db_utils

if __name__ == '__main__':
    # start the database
    db_utils.start_if_not_running()

    # connect to the database
    conn = db_utils.connect()

    # fetch a list of labels
    items = label.fetch_list(conn)
    print(items)
