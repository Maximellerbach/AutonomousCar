
def fetch_number_items(conn):
    sql = """select count(id)
        from lb_images
        """

    cursor = conn.cursor()
    cursor.execute(sql)
    return cursor.fetchone()[0]


def add(conn, image_dict):
    sql = """
        insert into lb_images(
            id, image)
        values({d[time]}, {d[image]});
        """
    cursor = conn.cursor()
    cursor.execute(sql.format(d=image_dict))
    return cursor.fetchone()


def fetch_list(conn, offset=0, limit=100):
    sql = """
        select *
        from lb_images
        order by id
        limit {d[limit]}
        offset {d[offset]}
        """

    cursor = conn.cursor()
    cursor.execute(sql.format(d={'offset': offset, 'limit': limit}))
    return cursor.fetchall()


def fetch_list_dataset(conn, dataset_name, offset=0, limit=100):
    sql = """
        select *
        from lb_images
        where dataset_name = {d[dataset_name]}
        order by id
        limit {d[limit]}
        offset {d[offset]}
        """

    cursor = conn.cursor()
    cursor.execute(sql.format(
        d={'dataset_name': dataset_name, 'offset': offset, 'limit': limit}))
    return cursor.fetchone()


def generator_fetch(conn, dataset_name=None, limit=100):
    n_items = fetch_number_items(conn)

    if dataset_name is None:
        for offset in range(0, n_items, limit):
            yield fetch_list(conn)
    else:
        for offset in range(0, n_items, limit):
            yield fetch_list_dataset(conn)
