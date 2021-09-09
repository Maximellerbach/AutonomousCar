def fetch_number_items(conn):
    sql = """
        select count(id)
        from lb_annotations
        """

    cursor = conn.cursor()
    cursor.execute(sql)
    return cursor.fetchone()[0]


def fetch_number_items_dataset(conn, dataset_name):
    sql = """
        select count(id)
        from lb_annotations
        where dataset_name = '{d[dataset_name]}'
        """

    cursor = conn.cursor()
    cursor.execute(sql.format(d={"dataset_name": dataset_name}))
    return cursor.fetchone()[0]


def fetch_rows_meta(conn):
    sql = """
        select column_name
        from information_schema.columns
        where table_name = 'lb_annotations'
    """
    cursor = conn.cursor()
    cursor.execute(sql)
    rows = cursor.fetchall()
    return [row[0] for row in rows]


def add(conn, annotation_dict):
    sql = """
        insert into lb_annotations(
            id, direction, speed, throttle, dataset_name, img_path)
        select {d[time]}, {d[direction]}, {d[speed]}, {d[throttle]}, '{d[dataset_name]}', '{d[img_path]}'

        where
        not exists (
            select id from lb_annotations where id = {d[time]}
        );
        """

    cursor = conn.cursor()
    cursor.execute(sql.format(d=annotation_dict))
    cursor.execute("COMMIT")


def add_list(conn, list_annotation_dict):
    sql = """
        insert into lb_annotations(
            id, direction, speed, throttle, dataset_name, img_path)
        select {d[time]}, {d[direction]}, {d[speed]}, {d[throttle]}, '{d[dataset_name]}', '{d[img_path]}'

        where
        not exists (
            select id from lb_annotations where id = {d[time]}
        );
        """
    cursor = conn.cursor()
    for annotation_dict in list_annotation_dict:
        cursor.execute(sql.format(d=annotation_dict))
    cursor.execute("COMMIT")


def fetch_by_id(conn, label_id):
    sql = """
        select *
        from lb_annotations
        where id = {d[label_id]}
    """
    cursor = conn.cursor()
    cursor.execute(sql.format(d={"label_id": label_id}))
    return cursor.fetchone()


def fetch_list(conn, offset=0, limit=100):
    sql = """
        select *
        from lb_annotations
        order by id
        limit {d[limit]}
        offset {d[offset]}
        """

    cursor = conn.cursor()
    cursor.execute(sql.format(d={"offset": offset, "limit": limit}))
    return cursor.fetchall()


def fetch_list_dataset(conn, dataset_name, offset=0, limit=100):
    sql = """
        select *
        from lb_annotations
        where dataset_name = '{d[dataset_name]}'
        order by id
        limit {d[limit]}
        offset {d[offset]}
        """

    cursor = conn.cursor()
    cursor.execute(sql.format(d={"dataset_name": dataset_name, "offset": offset, "limit": limit}))
    return cursor.fetchall()


def generator_load_dataset(conn, dataset_name, limit=100):
    n_items = fetch_number_items_dataset(conn, dataset_name)

    for offset in range(0, n_items, limit):
        yield fetch_list_dataset(conn, dataset_name, offset=offset, limit=limit)


def generator_load(conn, limit=100):
    n_items = fetch_number_items(conn)

    for offset in range(0, n_items, limit):
        yield fetch_list(conn, offset=offset, limit=limit)
