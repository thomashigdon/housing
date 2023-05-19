import sqlite3
import csv

def get_column_types(csv_file):
    """Infers the types of each column in a CSV file"""
    with open(csv_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        headers = next(reader)
        types = ['text' for _ in headers]
        for row in reader:
            for i, value in enumerate(row):
                try:
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

def insert_data(cursor, table_name, csv_file):
    """Inserts data from a CSV file into an SQLite table"""
    with open(csv_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        headers = next(reader)
        joined_headers = ', '.join(headers)
        values = ', '.join(['?' for _ in headers])
        query = "INSERT INTO {} ({}) VALUES ({})".format(table_name, joined_headers, values)
        for row in reader:
            cursor.execute(query, row)

if __name__ == '__main__':
    # Define input/output filenames and table name
    csv_file = sys.argv[1]
    db_file = sys.argv[2]
    table_name = sys.argv[3]

    # Get column types and create SQLite database and table
    types = get_column_types(csv_file)
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    delimiter = ','
    if 'tsv' in csv_file:
        delimiter = '\t'
    headers = next(csv.reader(open(csv_file), delimiter=delimiter))
    create_table(c, table_name, headers, types)

    # Insert data into table
    insert_data(c, table_name, csv_file)

    # Commit changes and close connection
    conn.commit()
    conn.close()

