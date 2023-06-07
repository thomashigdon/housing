import sqlite3
import sys
import argparse
from tabulate import tabulate

def main():
    parser = argparse.ArgumentParser(description="check comps")
    parser.add_argument("db", type=str)
    parser.add_argument("zpid", type=str)
    args = parser.parse_args()

    with sqlite3.connect(args.db) as conn:
        print("Comps for: ")
        cursor = conn.cursor()
        original_query = """
select zpid,
       ppsf,
       numberOfUnitsTotal,
       address_zipcode,
       hdpUrl
from properties
where zpid = ?
"""
        cursor.execute(original_query, (args.zpid,))
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        result = dict(zip(columns, rows[0]))
        zip_code = result['address_zipcode']

        # Format and print the result
        print(tabulate(rows, headers=columns, tablefmt='simple'))

        comp_query = """
select zpid,
       ppsf,
       hdpUrl,
       sqrt(pow(((select latitude from properties where zpid = ?) - latitude), 2) + pow(((select longitude from properties where zpid = ?) - longitude), 2)) * 5000 as distance,
       julianDay('now') - julianDay(soldDate) as daysSinceSold,
       soldDate,
       sqrt(pow(((select latitude from properties where zpid = ?) - latitude), 2) + pow(((select longitude from properties where zpid = ?) - longitude), 2)) * 5000 + julianDay('now') - julianDay(soldDate) as compIndex
from properties
where homeStatus = 'RECENTLY_SOLD' and
      homeType = 'MULTI_FAMILY'
order by compIndex asc
limit 10;
"""
        cursor.execute(comp_query, (args.zpid, args.zpid, args.zpid, args.zpid))
        rows = cursor.fetchall()
        # Get the column names
        columns = [description[0] for description in cursor.description]

        # Format and print the result
        print("\n\nRecently sold: ")
        print(tabulate(rows, headers=columns, tablefmt='simple'))

        rent_query = """
SELECT 
  cast(bedrooms as integer) bedrooms,
  cast(AVG(price) as integer) AS average_price,
  cast((
    SELECT price
    FROM (
      SELECT price,
      ROW_NUMBER() OVER (ORDER BY price) AS row_num,
      COUNT(*) OVER () AS total_rows
      FROM rentals AS r2
      WHERE r1.bedrooms = r2.bedrooms AND r1.address_zipcode = r2.address_zipcode
    ) AS subquery
    WHERE row_num IN ((total_rows + 1) / 2, (total_rows + 2) / 2)
    ORDER BY row_num
    LIMIT 1
  ) as integer) AS median_price,
  count(1) as number
FROM rentals AS r1
WHERE address_zipcode = ?
GROUP BY bedrooms;
"""
        cursor.execute(rent_query, (zip_code,))
        rows = cursor.fetchall()
        # Get the column names
        columns = [description[0] for description in cursor.description]

        # Format and print the result
        print("\n\nFor rent same zip code: ")
        print(tabulate(rows, headers=columns, tablefmt='simple'))
        cursor.close()

if __name__ == "__main__":
    sys.exit(main())
