import sqlite3
import csv
import sys

def get_column_types(csv_file, headers, delimiter='\t'):
    """Infers the types of each column in a CSV file"""
    with open(csv_file, 'r') as f:
        reader = csv.reader(f, delimiter=delimiter)
        next(reader)
        types = ['text' for _ in headers]
        for row in reader:
            limit = len(headers)
            for i, value in list(enumerate(row))[:limit]:
                try:
                    if 'name' in headers[i] or 'Name' in headers[i]:
                        continue
                    if '.' in value:
                        float(value)
                        types[i] = 'real'
                    else:
                        int(value)
                        types[i] = 'INTEGER'
                except ValueError:
                    pass
        return types

def create_table(cursor, table_name, headers, types):
    results = ', '.join(["{} {}".format(h, t) for h, t in zip(headers, types)])
    query = "CREATE TABLE {} ({})".format(table_name, results)
    print(query)
    cursor.execute(query)

def insert_data(cursor, table_name, csv_headers, sql_headers, csv_file, delimiter='\t'):
    """Inserts data from a CSV file into an SQLite table"""
    with open(csv_file, 'r') as f:
        reader = csv.reader(f, delimiter=delimiter)
        next(reader)
        values = ', '.join(['?' for _ in sql_headers])
        query = "INSERT INTO {} VALUES ({})".format(table_name, values)
        for row in reader:
            for i, r in enumerate(row):
                if i > len(sql_headers):
                    if row[i] == '': continue
                    cursor.execute(query, tuple(row[:len(sql_headers) - 2] + [csv_headers[i], row[i]]))
                elif i < len(sql_headers) and sql_headers[i] == 'RegionName':
                    row[i] = row[i].zfill(5)

if __name__ == '__main__':
    # Define input/output filenames and table name
    csv_file = sys.argv[1]
    db_file = sys.argv[2]
    table_name = sys.argv[3]

    delimiter = ','
    if 'tsv' in csv_file:
        delimiter = '\t'
    # Get column types and create SQLite database and table
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    csv_headers = next(csv.reader(open(csv_file), delimiter=delimiter))
    sql_headers = [h for h in csv_headers if not h[0].isdigit()]
    types = get_column_types(csv_file, sql_headers, delimiter=delimiter)
    for col, typ in [('Date', 'datetime'), ('Rent', 'real')]:
        sql_headers.append(col)
        types.append(typ)
    create_table(c, table_name, sql_headers, types)

    # Insert data into table
    insert_data(c, table_name, csv_headers, sql_headers, csv_file, delimiter=delimiter)

    # Commit changes and close connection
    conn.commit()
    conn.close()

