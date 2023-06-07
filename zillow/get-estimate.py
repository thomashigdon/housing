import sqlite3
import sys
import argparse
from tabulate import tabulate

def main():
    parser = argparse.ArgumentParser(description="check comps")
    parser.add_argument("db", type=str)
    args = parser.parse_args()

    with sqlite3.connect(args.db) as conn:
        cursor = conn.cursor()
        drop_statement = """
drop table if exists properties_instrumented;
"""
        cursor.execute(drop_statement)
        original_query = """
create table properties_instrumented as
with rents_by_size as (SELECT
  address_zipcode,
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
  GROUP BY bedrooms, address_zipcode),
temp_split_rows AS
  (WITH RECURSIVE cte_split_rows AS (
    SELECT zpid,
           substr(bedroomList, 1, instr(bedroomList || ',', ',') - 1) AS bedroom,
           substr(bedroomList, instr(bedroomList || ',', ',') + 1) AS remaining
    FROM properties
    UNION ALL
    SELECT zpid,
           substr(remaining, 1, instr(remaining || ',', ',') - 1) AS bedroom,
           substr(remaining, instr(remaining || ',', ',') + 1)
    FROM cte_split_rows
    WHERE remaining <> ''
  )
  SELECT zpid, bedroom  FROM cte_split_rows),

calculatedRents as (SELECT p.*,
       sum(rbs.median_price) as total_rent,
       (case
         when bedroomList like '%2%' then sum(rbs.median_price) -
                 (select median_price
                  from rents_by_size
                  where address_zipcode = p.address_zipcode and
                         bedrooms = 2)
         when bedroomList like '%3%' then sum(rbs.median_price) -
                 (select median_price
                  from rents_by_size
                  where address_zipcode = p.address_zipcode and
                         bedrooms = 3)
         when bedroomList like '%4%' then sum(rbs.median_price) -
                 (select median_price
                  from rents_by_size
                  where address_zipcode = p.address_zipcode and
                         bedrooms = 4)
         when bedroomList like '%5%' then sum(rbs.median_price) -
                 (select median_price
                  from rents_by_size
                  where address_zipcode = p.address_zipcode and
                         bedrooms = 5)
         else null end) as subtracted_rent
from properties p
inner join temp_split_rows tsr
  on tsr.zpid = p.zpid
inner join rents_by_size rbs
  on p.address_zipcode = rbs.address_zipcode and
     tsr.bedroom = rbs.bedrooms
group by p.zpid)
select *,
         subtracted_rent * (.97) /* occupancy */
         - mortgagePayment
         - estimatedMonthlyTax
         - estimatedMaintenance
         - 300 /* insurance */ as cashFlow,
         total_rent * (.97) /* occupancy */
         - mortgagePayment
         - estimatedMonthlyTax
         - estimatedMaintenance
         - 300 /* insurance */ as cashFlowAfterMoveout
from calculatedRents
order by cashFlow desc
"""
        cursor.execute(original_query)
        new_query = """
select * from properties_instrumented
"""
        cursor.execute(new_query)
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        result = dict(zip(columns, rows[0]))
        zip_code = result['address_zipcode']

        # Format and print the result
        print(tabulate(rows, headers=columns, tablefmt='simple'))

        cursor.close()

if __name__ == "__main__":
    sys.exit(main())
