import sqlite3

# connect to the database
conn = sqlite3.connect('housing.db')
c = conn.cursor()

# read the column definitions from the text file
with open('columns.txt') as f:
    column_defs = [line.strip().split() for line in f]

# get the current column names and types
c.execute('PRAGMA table_info(housing)')
result = c.fetchall()
current_columns = [row[1] for row in result]
current_types = [row[2] for row in result]
import pdb; pdb.set_trace()

# create a list of new column names and types
new_columns = current_columns[:]
new_types = current_types[:]
for col_def in column_defs:
    col_name, col_type = col_def
    for i, (col, type) in enumerate(zip(new_columns, new_types)):
        if col == col_name:
            new_types[i] = col_type

# create a temporary table with the new column definitions
temp_table_name = 'temp_housing'
string = f'CREATE TABLE {temp_table_name} ({", ".join([f"{col_name} {col_type}" for col_name, col_type in zip(new_columns, new_types)])})'
import pdb; pdb.set_trace()
c.execute(f'CREATE TABLE {temp_table_name} ({", ".join([f"{col_name} {col_type}" for col_name, col_type in zip(new_columns, new_types)])})')

# copy data from the old table to the temporary table
c.execute(f'INSERT INTO {temp_table_name} SELECT {", ".join(current_columns))} FROM housing')

# drop the old table and rename the temporary table to the original name
c.execute('DROP TABLE housing')
c.execute(f'ALTER TABLE {temp_table_name} RENAME TO housing')

# commit the changes and close the connection
conn.commit()
conn.close()
