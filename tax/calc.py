import argparse
import csv
import pprint
from re import sub
import re
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import sqlite3
import sys
from collections import namedtuple
from matplotlib.widgets import CheckButtons
from matplotlib.widgets import RectangleSelector
from data import zip_codes_of_interest

import mplcursors

cpi = {
 2000:168.8,
 2001:175.1,
 2002:177.1,
 2003:181.7,
 2004:185.2,
 2005:190.7,
 2006:198.3,
 2007:202.416,
 2008:211.080,
 2009:211.143,
 2010:216.687,
 2011:220.223,
 2012:226.665,
 2013:230.280,
 2014:233.916,
 2015:233.707,
 2016:236.916,
 2017:242.839,
 2018:247.867,
 2019:251.712,
 2020:257.761,
 2021:261.582,
 2022:281.148,
 2023:299.170,
}

dividend_yield = .0175
dividend_yield_py = {
 2020: .0158,
 2019: .0183,
 2018: .0209,
 2017: .0184,
 2016: .0203,
 2015: .0211,
 2014: .0192,
 2013: .0194,
 2012: .0220,
 2011: .0213,
}

# https://themortgagereports.com/61853/30-year-mortgage-rates-chart
# https://fred.stlouisfed.org/series/MORTGAGE30US
mortgage_rates = {
 2011:.0445,
 2012:.0366,
 2013:.0398,
 2014:.0417,
 2015:.0385,
 2016:.0365,
 2017:.0399,
 2018:.0454,
 2019:.0394,
 2020:.0310,
 2021:.0296,
 2022:.0534,
 2023:.0633,
}

def calc_income_needed_for_monthly_payment(payment, rule_of_thumb_percentage):
    return payment * 12 / rule_of_thumb_percentage

def open_zipcode_file(name, newline='\r\n'):
    lines = []
    with open(name, newline=newline) as f:
      blah = csv.reader(f, dialect='excel')
      for b in blah:
            lines.append(b)
    return lines

def process_lines(lines):
    data = {}
    headers = lines[3]
    for i, header in enumerate(headers):
        headers[i] = header.replace("\n", " ")
    for row in lines[4:]:
        if not row[0]: continue

        zip_code = row[0]
        income_bin = (row[1].replace('$1 under $25,000', '025000')
                            .replace('$25,000 under $50,000', '050000')
                            .replace('$50,000 under $75,000', '075000')
                            .replace('$75,000 under $100,000', '100000')
                            .replace('$100,000 under $200,000', '200000')
                            .replace('$200,000 or more', 'plus'))

        # Iterate over each column in the row
        for i in range(2, len(headers)):
            if 'Adjusted gross income' not in headers[i] and 'Number of returns' not in headers[i] and 'Ordinary dividends'  not in headers[i]: continue
            reps = [(r"Number of returns( \[2\])?", "num"),
                    (r"Adjusted gross income.*", "agi"),
                    (r"Ordinary dividends.*", "div"),
                    ]
            my_header = headers[i]
            for regex, rep in reps:
                my_header = sub(regex, rep, my_header)
            col = row[i].translate(str.maketrans("", "", ",\n"))
            if my_header == 'div':
                # dividends are in the next column. This column has the number of reporters
                col = row[i+1].translate(str.maketrans("", "", ",\n"))
            try:
                col = int(col)
            except ValueError:
                pass
            if zip_code not in data: data[zip_code] = {}
            if income_bin not in data[zip_code]: data[zip_code][income_bin] = {}
            data[zip_code][income_bin][my_header] = col
    return data

def all_valid(data_zip):
    return all([x['num'] != '** ' for x in data_zip.values()])

def get_percentile(pct, in_data, zip_code, item_keys):
    def percentile_general(pct, in_data, zip_code):
        for i, key in enumerate(item_keys[:-1]):
            item = in_data[zip_code][item_keys[i]]
            item_plus_key = item_keys[i+1]
            item_plus = in_data[zip_code][item_plus_key]
            # print("pct target: ", pct, "pct: ", item['pct'], "pct_plus: ", item_plus['pct'])
            if not 'pct' in item or not 'pct' in item_plus: continue
            if item['pct'] <= pct <= item_plus['pct']:
                # this is our item
                # print("detected range {} for zip {}".format(item_plus_key, zip))
                base_pct = item_plus['pct'] - item['pct']
                base_num = int(item_plus_key) - int(item_keys[i])
                into_range = (pct-item['pct']) / base_pct
                into_num = into_range * base_num
                actual_num = into_num + int(item_keys[i])
                return actual_num
        def percentile_upper(pdes, pct, pareto):
            pct_within_bound = (pdes - pct) / (1-pct)
            upper_bound = int(item_keys[-1])
            return ((upper_bound ** pareto) / (1 - pct_within_bound)) ** (1/pareto)
        if not 'pareto' in in_data[zip_code]['plus']: return 0
        return percentile_upper(pct, in_data[zip_code]['200000']['pct'], in_data[zip_code]['plus']['pareto'])
    return percentile_general(pct, in_data, zip_code)

def get_percentile_for_income(income, in_data, year, zip_code, item_keys):
    if not year in in_data or not zip_code in in_data[year] or not income: return None
    data = in_data[year][zip_code]['raw']
    base = xm = int(item_keys[-1])
    for i, key in enumerate(item_keys[:-1]):
        item = data[item_keys[i]]
        item_plus_key = item_keys[i+1]
        item_plus = data[item_plus_key]
        if not 'pct' in item or not 'pct' in item_plus: continue
        if int(key) <= income <= int(item_plus_key):
            # this is our item
            base_pct = item_plus['pct'] - item['pct']
            base_income = int(item_plus_key) - int(item_keys[i])
            into_range = (income - int(item_keys[i])) / base_income
            into_pct = into_range * base_pct
            actual_pct = into_pct + item['pct']
            return actual_pct
    if not 'pareto' in data['plus']: return 0
    pct_raw = 1 - (xm/income) ** data['plus']['pareto']
    base_pct = data[item_keys[-1]]['pct']
    pct_range_size = 1 - base_pct
    return base_pct + pct_raw * pct_range_size

def process_data(data, zips, year):
    out = {}
    if not zips:
        zips = [zip for zip in data.keys() if len(zip) == 5 and zip != '00000']
    zips = [zip for zip in zips if all_valid(data[zip])]
    for zip in zips:
        total = 0
        upper_bound = 200e3
        for key in sorted(data[zip].keys()):
            if data[zip][key]['agi'] in ('** ', '**') or data[zip][key]['num'] in ('** ', '**'): continue
            try:
                data[zip][key]['avg_agi'] = data[zip][key]['agi'] / data[zip][key]['num'] * 1000
            except ZeroDivisionError:
                continue
            if not data[zip][key]['div'] in ('**', '** '):
                data[zip][key]['wealth_inv'] = int(round(data[zip][key]['div'] / dividend_yield_py[year] / data[zip][key]['num'] * 1000, -2))
            else:
                data[zip][key]['wealth_inv'] = 0
            if not key: continue
            total += data[zip][key]['num']
            data[zip][key]['cum'] = total

        for key in sorted(data[zip].keys()):
            if not key or data[zip][key]['agi'] in ('** ', '**', 0) or data[zip][key]['num'] in ('** ', '**'): continue
            if not 'cum' in data[zip][key]: continue
            data[zip][key]['pct'] = data[zip][key]['cum'] / total

        item = data[zip]['plus']
        if 'avg_agi' in item:
            item['pareto'] = item['avg_agi'] / (item['avg_agi'] - upper_bound)

    for zip_code in zips:
        item_keys = ['025000', '050000', '075000', '100000', '200000']
        out[zip_code] = {}
        for pct in ['50', '75', '90', '95', '99']:
            key = "p" + pct
            out[zip_code][key] = int(round(get_percentile(float(pct)/100, data, zip_code, item_keys), -2))
        if zip_code in zip_codes_of_interest:
            out[zip_code]['location'] = zip_codes_of_interest[zip_code]
        if 'avg_agi' in data[zip_code]['']:
            out[zip_code]['avg'] = round(int(data[zip_code]['']['avg_agi']), -2)
        out[zip_code]['wealth'] = {}
        out[zip_code]['avg_wealth'] = data[zip_code][''].get('wealth_inv', 0)
        out[zip_code]['wealth-100k-200k'] = data[zip_code]['200000'].get('wealth_inv', 0)
        out[zip_code]['wealth-200k-plus'] = data[zip_code]['plus'].get('wealth_inv', 0)
        out[zip_code]['num'] = {}
        for key in [''] + item_keys + ['plus']:
            out[zip_code]['wealth'][key] = data[zip_code][key].get('wealth_inv', 0)
            out[zip_code]['num'][key] = data[zip_code][key].get('num', 0)
        out[zip_code]['raw'] = data[zip_code]
    return out

def get_y_values_income(income_data, years, zip_codes, whats):
    def wtag(what, zip_code):
        tag = what + " income"
        if "wealth" in what:
            tag = what
        return tag + '-' + zip_name(zip_code)
    y_values = []
    for what in whats:
        for zip_code in zip_codes:
            what_values = []
            for year in years:
                what_values.append(income_data[year][zip_code][what])
            y_values.append([what_values, wtag(what, zip_code)])
    return y_values

def line_plot_income(income_data, zip_codes, whats):
    # Extract x and y values from data
    x_values = list(income_data.keys())
    y_values = get_y_values_income(income_data, x_values, zip_codes, whats)

    # Create a new figure and set its size
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the data as a line chart
    for y_value in y_values:
        ax.plot(x_values, y_value[0], marker='o', label=y_value[1])

    # Set the title and axis labels
    ax.set_title('Income in ZIP codes by year')
    ax.set_xlabel('Year')
    ax.set_ylabel('Income')

    if len(y_values) < 10:
        ax.legend()
    mplcursors.cursor(hover=True).connect("add", lambda sel: sel.annotation.set_text(f"Series Name: {sel.artist.get_label()}, Value: {sel.target[1]}"))

    # Show the plot
    plt.show()

def zip_name(zip_code):
    if zip_code in zip_codes_of_interest:
        return zip_codes_of_interest[zip_code] + '-' + zip_code
    return zip_code

def get_y_values_housing(hdata, years, zip_codes, property_types, whats):
    y_values = []
    def get_label(property_type, what, zip_code):
        if len(zip_codes) > 1:
            return property_type + '-' + what + '-' + zip_name(zip_code)

        return zip_codes_of_interest.get(zip_code, zip_code)
    for what in whats:
        if what == 'rent': continue
        for property_type in property_types:
            for zip_code in zip_codes:
                what_values = []
                for year in years:
                    if year not in hdata:
                        what_values.append(None)
                        continue
                    income_data = hdata[year]
                    if zip_code not in income_data or property_type not in income_data[zip_code]:
                        what_values.append(None)
                        continue
                    what_values.append(income_data[zip_code][property_type][what])
                y_values.append([what_values, get_label(property_type, what, zip_code)])
    return y_values

def get_y_values_rent(rdata, years, zip_codes):
    y_values = []
    if not rdata:
        return y_values
    for zip_code in zip_codes:
        rent_values = []
        for year in years:
            if year not in rdata:
                rent_values.append(None)
                continue
            income_data = rdata[year]
            if not zip_code in income_data:
                rent_values.append(None)
                continue
            rent_values.append(income_data[zip_code]['rent'])
        y_values.append([rent_values, 'rent' + '-' + zip_name(zip_code)])
    return y_values

def line_plot_combined(hdata, income_data, rdata, zip_codes, property_types, whats_income, whats_housing):
    # Extract x and y values from data
    x_values = sorted(list(set(list(hdata.keys()) + list(income_data.keys()))))
    y_values_housing = get_y_values_housing(hdata, x_values, zip_code, property_types, whats_housing)
    y_values_income = get_y_values_income(income_data, x_values, zip_codes, whats_income)
    y_values_rent = get_y_values_rent(rdata, x_values, zip_code)

    # normalize each to 2015 = 1
    base_index = x_values.index(2015)
    for thing in [y_values_housing, y_values_income, y_values_rent]:
        for i, y_value in enumerate(thing):
            base = y_value[0][base_index]
            if base == None:
                # remove the 
                print("Warning removing {} line because we don't have baseline data".format(y_value[1]))
                thing.pop(i)
                continue
            y_value[0] = [x/base if x != None else x for x in y_value[0]]
            y_value[1] = y_value[1] + " " + "2015 = {}".format(int(base))

    # Create a new figure and set its size
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Plot the data as a line chart
    for y_value in y_values_housing:
        ax1.plot(x_values, y_value[0], marker='o', label=y_value[1])

    for y_value in y_values_income:
        ax1.plot(x_values, y_value[0], marker='x', label=y_value[1])

    for y_value in y_values_rent:
        ax1.plot(x_values, y_value[0], marker='+', label=y_value[1])

    ax1.legend()
    #ax1.set_ylim([0, None])
    # Set the title and axis labels
    ax1.set_title('{} data in {} by year'.format(', '.join(property_types), zip_codes_of_interest[zip_code]))
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Homes (dollars)')
    ax1.set_ylabel('Income/wealth (dollars)')

    # Show the plot
    plt.show()

def line_plot_housing(hdata, rdata, zip_codes, property_types, whats):
    # Extract x and y values from data
    x_values = sorted(list(hdata.keys()))
    #y_values = [([getattr(x[zip_code][property_type] if property_type in x[zip_code] else None, what) for x in hdata.values()], what) for what in whats]
    y_values = get_y_values_housing(hdata, x_values, zip_codes, property_types, whats)
    if 'rent' in whats:
        y_values += get_y_values_rent(rdata, x_values, zip_codes)
    # Create a new figure and set its size
    fig, ax = plt.subplots(figsize=(16, 9))

    # Plot the data as a line chart
    for y_value in y_values:
        ax.plot(x_values, y_value[0], marker='o', label=y_value[1])

    #plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), ncol=1)
    # Set the title and axis labels
    ax.set_title('{} data by year'.format(', '.join(property_types)))
    ax.set_xlabel('Year')
    ax.set_ylabel('Dollars')
    ax.set_ylim([0, None])

    if len(y_values) < 10:
        ax.legend()
    mplcursors.cursor(hover=True).connect("add", lambda sel: sel.annotation.set_text(f"Series Name: {sel.artist.get_label()}"))

    # Create a callback function for the rectangle selector
    def onselect(eclick, erelease):
        # Get the coordinates of the selected box
        x1, x2 = sorted([eclick.xdata, erelease.xdata])
        y1, y2 = sorted([eclick.ydata, erelease.ydata])
        # Set the axis limits to the selected box
        ax.set_xlim(x1, x2)
        ax.set_ylim(y1, y2)
        # Redraw the plot
        fig.canvas.draw()

    # Create the rectangle selector
    rect_selector = RectangleSelector(ax, onselect, drawtype='box', interactive=True)


    # Show the plot
    plt.show()

def map_plot(zip_code_data):
    zipcodes = gpd.read_file('mass.shp')
    zip_col = 'ZCTA5CE10'

    zlist = list(zip_code_data)
    # Merge your own data with the zip code dataframe
    data = {'zipcode': [x for x,y in zlist], 'value': [y for (x,y) in zlist]}
    data_df = pd.DataFrame(data)
    zipcodes = zipcodes.merge(data_df, left_on=zip_col, right_on='zipcode')

    # Visualize the map with color-coding
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_axis_off()
    ax.set_aspect(1)

    zipcodes.plot(column='value', cmap='OrRd', ax=ax, legend=True)
    plt.title('Boston Zip Codes')
    for idx, row in zipcodes.iterrows():
        ax.annotate(text=row[zip_col], xy=row['geometry'].centroid.coords[0], ha='center', fontsize=8)
    plt.gcf().canvas.toolbar.pan()

    plt.show()

def inflation_adjust_income(data, year):
    cur_cpi = cpi[2023]
    old_cpi = cpi[year]

    for k, v in data.items():
        if 'num' in k or type(v) == str or 'raw' in k: continue
        elif type(v) == dict:
            for ink, inv in v.items():
                v[ink] = inv * (cur_cpi / old_cpi)
        else:
            data[k] = v * (cur_cpi / old_cpi)

def inflation_adjust_housing(data, year):
    cur_cpi = cpi[2023]
    old_cpi = cpi[year]

    for k, v in data.items():
        if 'pct' in k: continue
        if 'ppsf' in k or 'price' in k or 'payment' in k and v is not None:
            data[k] = v * (cur_cpi / old_cpi)

def inflation_adjust_rent(data, year):
    cur_cpi = cpi[2023]
    old_cpi = cpi[year]

    for zip_code, item in data.items():
        for k, v in item.items():
            if k == 'rent':
                item[k] = v * (cur_cpi / old_cpi)

def get_mortgage_payment(year, price, rate=None, multiplier=1, yearly_multiplier=1):
    P = 0.8 * price
    # rate divided by 12 for monthly rate
    i = (mortgage_rates[year] if not rate else rate) / 12
    n = 360

    extra_payment = price * multiplier * (yearly_multiplier - 1) / 12
    result = P * i / (1-(1+i)**(-n)) * multiplier + extra_payment
    return result

def get_housing(income_data):
    # Connect to the database
    conn = sqlite3.connect('housing.db')

    # Create a cursor object
    cur = conn.cursor()

    # Execute a SELECT query
    cur.execute("""
SELECT strftime('%Y', period_begin) as year, substr(region, instr(region, ':') + 2) as zip_code,
         (case when property_type = 'Single Family Residential' then 'sfh'
               when property_type in ('Townhouse', 'Condo/Co-op') then 'condo'
               when property_type in ('All Residential') then  'all'
               when property_type in ('Multi-Family (2-4 Unit)') then 'mfh'
               else property_type
        end) as type,
avg(median_sale_price) as median_sale_price,
avg(median_list_price) as median_list_price,
avg(median_ppsf) as median_ppsf,
avg(median_list_ppsf) as median_list_ppsf,
sum(homes_sold) as homes_sold,
sum(pending_sales) as pending_sales,
sum(new_listings) as new_listings,
sum(inventory) as inventory,
avg(months_of_supply) as months_of_supply,
avg(median_dom) as median_dom,
avg(avg_sale_to_list) as avg_sale_to_list,
avg(sold_above_list) as sold_above_list,
sum(price_drops) as price_drops,
avg(off_market_in_two_weeks) as off_market_in_two_weeks
FROM housing
WHERE strftime('%m', period_begin) in ('01', '04', '07', '10')
GROUP BY 1, 2, 3
ORDER BY 1, 2, 3 ASC
""")

    # Fetch all the rows
    rows = cur.fetchall()

    column_names = [description[0] for description in cur.description]
    rows = [dict(zip(column_names, row)) for row in rows]

    data = {}
    for row in rows:
        year = int(row['year'])
        zip_code = row['zip_code']
        if year not in data:
            data[year] = {}
        if zip_code not in data[year]:
            data[year][zip_code] = {}
        property_type = row['type']
        row['median_mortgage_ppsf_norm'] = get_mortgage_payment(year, row['median_ppsf']) / 0.8 * 20

        # Mortgage payment at actual interest rates
        row['1000sf_mortgage_payment'] = get_mortgage_payment(year, row['median_ppsf'], multiplier=1000)
        row['1000sf_mortgage_payment_pct']  = get_percentile_for_income(
                calc_income_needed_for_monthly_payment(row['1000sf_mortgage_payment'], 0.4),
                                                       income_data, year, zip_code, item_keys)

        # 'pct' means the percentile of income needed to pay the mortgage using x% of income e.g. 40%

        # Mortgage payment including extra expenses
        row['1000sf_mortgage_payment_plus'] = get_mortgage_payment(year, row['median_ppsf'], multiplier=1000, yearly_multiplier=1.02)
        row['1000sf_mortgage_payment_plus_pct']  = get_percentile_for_income(
                calc_income_needed_for_monthly_payment(row['1000sf_mortgage_payment_plus'], 0.4),
                                                       income_data, year, zip_code, item_keys)

        # Mortgage payment at with 3% interest rate
        row['1000sf_3pct_mortgage_payment'] = get_mortgage_payment(year, row['median_ppsf'], rate=.03, multiplier=1000)
        row['1000sf_3pct_mortgage_payment_pct']  = get_percentile_for_income(
                calc_income_needed_for_monthly_payment(row['1000sf_3pct_mortgage_payment'], 0.4),
                                                       income_data, year, zip_code, item_keys)

        # Mortgage payment at 3% rate including extra expenses
        row['1000sf_3pct_mortgage_payment_plus'] = get_mortgage_payment(year, row['median_ppsf'], rate=.03, multiplier=1000, yearly_multiplier=1.02)
        row['1000sf_3pct_mortgage_payment_plus_pct']  = get_percentile_for_income(
                calc_income_needed_for_monthly_payment(row['1000sf_3pct_mortgage_payment_plus'], 0.4),
                                                       income_data, year, zip_code, item_keys)

        data[year][zip_code][property_type] = row


    # Close the cursor and connection
    cur.close()
    conn.close()
    return data

def get_rent(income_data):
    # Connect to the database
    conn = sqlite3.connect('housing.db')

    # Create a cursor object
    cur = conn.cursor()

    # Execute a SELECT query
    cur.execute("""
SELECT strftime('%Y', Date) as year, RegionName as zip_code, avg(Rent) as rent
FROM rent
GROUP BY 1, 2
ORDER BY 1, 2 ASC
""")

    # Fetch all the rows
    rows = cur.fetchall()

    column_names = [description[0] for description in cur.description]
    rows = [dict(zip(column_names, row)) for row in rows]

    data = {}
    for row in rows:
        year = int(row['year'])
        zip_code = row['zip_code']
        if year not in data:
            data[year] = {}
        if zip_code not in data[year]:
            data[year][zip_code] = {}
        data[year][zip_code] = row
        rent = row['rent']
        income_needed = calc_income_needed_for_monthly_payment(rent, 0.4)
        row['income_needed'] = income_needed
        if not year in income_data or not zip_code in income_data[year]: continue
        pct_needed = get_percentile_for_income(income_needed, income_data[year][zip_code]['raw'], year, zip_code, item_keys)
        row['income_pct_needed'] = pct_needed

    # Close the cursor and connection
    cur.close()
    conn.close()
    return data

#zips = zip_codes_of_interest
#zips = ['02118']
#zips = [zip for zip in data.keys() if len(zip) == 5 and zip != '00000']

item_keys = ['025000', '050000', '075000', '100000', '200000']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--line-plot-housing", help="line plot with housing data", action="store_true")
    parser.add_argument("--line-plot-income", help="line plot with income data", action="store_true")
    parser.add_argument("--zips", help="ZIP codes to plot", default=zip_codes_of_interest.keys(), nargs="+")
    parser.add_argument("whats", help="what to plot", nargs="+")
    parser.add_argument("--property-types", help="property types", default=["sfh", "condo", "mfh"], nargs="+")
    args = parser.parse_args()

    zip_codes = args.zips

    income_data = {}
    total_year_range = range(2011, 2022)
    for year in range(2011, 2021):
        filename = "{}zp22ma.csv".format(year - 2000)
        lines = open_zipcode_file(filename)
        data = process_lines(lines)
        income_data[year] = process_data(data, zip_codes, year)

    hdata = get_housing(income_data)
    rdata = get_rent(income_data)

    for year in total_year_range:
        if not year in rdata: continue
        rdy = rdata[year]
        for zip_code, data in rdy.items():
            rent = data['rent']
            income_needed = calc_income_needed_for_monthly_payment(rent, 0.4)
            data['income_needed'] = income_needed
            if not year in income_data or not zip_code in income_data[year]: continue
            pct_needed = get_percentile_for_income(income_needed, income_data[year][zip_code]['raw'], year, zip_code, item_keys)
            data['pct_needed'] = pct_needed

    for year, item in income_data.items():
        for zip_code, data in item.items():
            inflation_adjust_income(data, year)

    for year, item in hdata.items():
        for _, pptype in item.items():
            for property_type, data in pptype.items():
                inflation_adjust_housing(data, year)

    for year, item in rdata.items():
        inflation_adjust_rent(item, year)

    whats = args.whats
    property_types = args.property_types
    #line_plot_combined(hdata, income_data, zip_code, ['all', 'condo', 'sfh', 'mfh'], ['p75', 'p90'], [])
    #line_plot_combined(hdata, income_data, rdata, zip_code, ['mfh'], ['p75'], whats)
    if args.line_plot_housing:
        line_plot_housing(hdata, rdata, zip_codes, property_types, whats)

    if args.line_plot_income:
        line_plot_income(income_data, zip_codes, whats)

    #map_plot(zip(zip_codes_of_interest, [hdata[2022][x]['condo']['median_ppsf'] / hdata[2022][x]['mfh']['median_ppsf'] if ('mfh' in hdata[2022][x] and hdata[2022][x]['mfh']['homes_sold'] > 5) else None for x in zip_codes_of_interest]))
    #map_plot(zip(zip_codes_of_interest, [hdata[2020][x]['condo']['1000sf_mortgage_payment_plus_pct'] for x in zip_codes_of_interest]))

if __name__ == "__main__":
    sys.exit(main())
