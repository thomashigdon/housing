import argparse
import asyncio
import datetime
import json
from random import randint
from typing import List
from urllib.parse import urlencode, quote
import mortgage
import pprint
import time
import pickle
import re
import os
import sqlite3

from loguru import logger as log
from parsel import Selector
import requests
from data import zip_codes_of_interest, Location
import data

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36',
           'Accept': '*/*',
           'Accept-Encoding': 'gzip, deflate, br',
           'Accept-Language': 'en,es-US;q=0.9,es;q=0.8,es-419;q=0.7',
           }

import asyncio

import time

def rate_limiter(max_calls, period, burst):
    """
    Decorator that limits the rate at which a function can be called
    using a token bucket algorithm.

    Args:
        max_calls (int): Maximum number of calls allowed within the given period.
        period (float): Length of the period in seconds.
        burst (int): Number of calls that are allowed to exceed the rate limit.

    Returns:
        Callable: Decorated function.
    """
    # Initialize the token bucket.
    tokens = max_calls
    burst = max(burst, max_calls)

    # Record the last time the function was called.
    last_called = 0

    def decorate(func):
        async def wrapper(*args, **kwargs):
            nonlocal tokens, last_called

            while True:
                # Calculate how many tokens should be added to the bucket.
                now = time.time()
                elapsed = now - last_called
                refill = elapsed * max_calls / period

                new_tokens = min(tokens + refill, burst)
                # If the burst limit is exceeded, raise an exception.
                if new_tokens < 0:
                    await asyncio.sleep(.1)
                else:
                    tokens = new_tokens
                    break

            # Call the function and decrement the token count.
            tokens -= 1
            last_called = now
            return func(*args, **kwargs)

        return wrapper

    return decorate

def _convert_dict_to_tuple(d):
    """Recursively converts a dictionary to a tuple of sorted key-value pairs."""
    if isinstance(d, dict):
        return tuple((k, _convert_dict_to_tuple(v)) for k, v in sorted(d.items()))
    else:
        return d

import hashlib

def hash_args(*args):
    arg_str = ''.join(map(str, args))
    hash_digest = hashlib.sha256(arg_str.encode()).hexdigest()
    hex_int = int(hash_digest[:8], 16)
    return 2 + (hex_int % 9)  # map to range [2, 10]

def print_args(func):
    def wrapper(*args, **kwargs):
        print(f"calling Zillow: {args}")
        result = func(*args, **kwargs)
        return result
    return wrapper

def memoize(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            cache = pickle.load(f)
    else:
        cache = {}
    def memoize(func):
        async def wrapper(*args, **kwargs):
            memoize = kwargs.get('memoize', False)
            if 'memoize' in kwargs:
                kwargs.pop('memoize')
            cache_key = (_convert_dict_to_tuple(args), _convert_dict_to_tuple(kwargs))
            in_cache = memoize and cache_key in cache
            is_requests_response = (type(cache.get(cache_key, None)) == requests.models.Response)
            if in_cache and is_requests_response and cache[cache_key].ok:
                result = cache[cache_key]
            else:
                result = await func(*args, **kwargs)
                cache[cache_key] = result
                tmp_file = filename + '.tmp'
                with open(tmp_file, 'wb') as f:
                    pickle.dump(cache, f)
                os.rename(tmp_file, filename)
            return result
        return wrapper
    return memoize

m_requests_get =  memoize('cache.pkl')(rate_limiter(1, 1, 1)(print_args(requests.get)))

async def _search(zips: List[str], filters: dict = {}, categories=("cat1", "cat2"), memoize=True) -> List[dict]:
    """base search function which is used by sale and rent search functions"""
    #html_result = await session.async_scrape(
    #    ScrapeConfig(
    #        url=f"https://www.zillow.com/homes/{query}_rb/",
    #        #proxy_pool="public_residential_pool",
    #        #country="US",
    #        asp=True,
    #    )
    #)
    price_limit = 10e6
    default_query_data = {"mapBounds":{"north":42.44534296278287,"east":-70.97269326611328,"south":42.24767622824179,"west":-71.26039773388672},"isMapVisible":False,"filterState":{"sortSelection":{"value":"globalrelevanceex"},"isSingleFamily":{"value":False},"isTownhouse":{"value":False},"isCondo":{"value":False},"isLotLand":{"value":False},"isApartment":{"value":False},"isManufactured":{"value":False},"isApartmentOrCondo":{"value":False},"price":{"max":price_limit}},"isListVisible":True,"mapZoom":12,"category":"cat1","pagination":{}}

    query_data = default_query_data
    query_data["filterState"].update(filters)

    regions = {"regionSelection": [{"regionId": int(z) + 56514} for z in zips]}
    query_data.update(regions)

    url = "https://www.zillow.com/search/GetSearchPageState.htm?"
    found = []
    # cat1 - Agent Listings
    # cat2 - Other Listings
    is_recently_sold = "isRecentlySold" in query_data["filterState"] and query_data["filterState"]["isRecentlySold"]["value"]
    if is_recently_sold:
        categories = ["cat1"]
        query_data.pop("category")
    for category in categories:
        search_query_stats = json.dumps(query_data)
        #search_query_stats_alt = '{"mapBounds":{"north":42.403895,"east":-71.063562,"south":42.352141,"west":-71.118159},"mapZoom":13,"isMapVisible":false,"filterState":{"doz":{"value":"30"},"isCondo":{"value":false},"isForSaleForeclosure":{"value":false},"isApartment":{"value":false},"sortSelection":{"value":"globalrelevanceex"},"isAuction":{"value":false},"isNewConstruction":{"value":false},"isRecentlySold":{"value":true},"isSingleFamily":{"value":false},"isLotLand":{"value":false},"isTownhouse":{"value":false},"isManufactured":{"value":false},"isForSaleByOwner":{"value":false},"isComingSoon":{"value":false},"isApartmentOrCondo":{"value":false},"isForSaleByAgent":{"value":false}},"isListVisible":true,"regionSelection":[{"regionId":58657},{"regionId":58659},{"regionId":58655},{"regionId":58656},{"regionId":58653}],"pagination":{}}'
        if is_recently_sold:
            wants = json.dumps({"cat1": ["listResults"], "regionResults":["regionResults"]})
        else:
            wants = json.dumps({category: ["mapResults"]})
        full_query = {
            "searchQueryState": search_query_stats,
            "wants": wants,
            "requestId": hash_args(search_query_stats, wants),
        }
        #api_result = await session.async_scrape(
        #    ScrapeConfig(
        #        url=url + urlencode(full_query, quote_via=quote),
        #        #proxy_pool="public_residential_pool",
        #        #country="US",
        #        asp=True,
        #    )
        #)
        response = await m_requests_get(url + urlencode(full_query, quote_via=quote), headers=headers, memoize=memoize)
        data = json.loads(response.content)
        _total = data["categoryTotals"][category]["totalResultCount"]
        if _total > 500:
            log.warning(f"query has more results ({_total}) than 500 result limit ")
        else:
            log.info(f"found {_total} results for zips: {zips}")
        if is_recently_sold:
            map_results = data[category]["searchResults"]['listResults']
        else:
            map_results = data[category]["searchResults"]["mapResults"]
        found.extend(map_results)
    return found


async def search_sale(zips: List[str],  memoize=True) -> List[dict]:
    """search properties that are for sale"""
    return await _search(zips=zips,  memoize=memoize)

async def search_sold(zips: List[str],  memoize=True) -> List[dict]:
    """search properties that are for sale"""
    filters = {
        "doz":{"value":"360"},
        "isRecentlySold": {"value": True},
        'isApartment': {'value': False},
        'isApartmentOrCondo': {'value': False},
        'isAuction': {'value': False},
        'isComingSoon': {'value': False},
        'isCondo': {'value': False},
        'isForSaleByAgent': {'value': False},
        'isForSaleByOwner': {'value': False},
        'isForSaleForeclosure': {'value': False},
        'isLotLand': {'value': False},
        'isManufactured': {'value': False},
        'isNewConstruction': {'value': False},
    }
    return await _search(zips=zips, filters=filters, memoize=memoize)

async def search_rent(zips: List[str], memoize=True) -> List[dict]:
    """search properites that are for rent"""
    log.info(f"scraping rent search for: {zips}")
    filters = {
        "isForSaleForeclosure": {"value": False},
        "isMultiFamily": {"value": False},
        "isAllHomes": {"value": True},
        "isAuction": {"value": False},
        "isNewConstruction": {"value": False},
        "isForRent": {"value": True},
        "isLotLand": {"value": False},
        "isManufactured": {"value": False},
        "isForSaleByOwner": {"value": False},
        "isComingSoon": {"value": False},
        "isForSaleByAgent": {"value": False},
    }
    return await _search(zips=zips, filters=filters, categories=["cat1"], memoize=memoize)


def parse_property(data: dict) -> dict:
    """parse zillow property"""
    # zillow property data is massive, let's take a look just
    # at the basic information to keep this tutorial brief:
    parsed = {
        "address": data["address"],
        "description": data["description"],
        "photos": [photo["url"] for photo in data["galleryPhotos"]],
        "zipcode": data["zipcode"],
        "phone": data["buildingPhoneNumber"],
        "name": data["buildingName"],
        # floor plans include price details, availability etc.
        "floor_plans": data["floorPlans"],
    }
    return parsed

async def gather_with_concurrency(n, *coros):
    semaphore = asyncio.Semaphore(n)

    async def sem_coro(coro):
        async with semaphore:
            return await coro
    return await asyncio.gather(*(sem_coro(c) for c in coros))

async def scrape_properties(urls: List[str], memoize=True):
    """scrape zillow properties"""

    async def scrape(url):
        #result = await session.async_scrape(
        #    ScrapeConfig(url=url, asp=True, country="US")
        #)
        #response = result.upstream_result_into_response()
        #api_response = response.text
        response = await m_requests_get(url, headers=headers, memoize=memoize)
        api_response = response.text
        sel = Selector(text=api_response)
        data = sel.css("script#__NEXT_DATA__::text").get()
        try:
            if data:
                # some properties are located in NEXT DATA cache
                data = json.loads(data)
                try:
                    data = json.loads(data["props"]["pageProps"]["gdpClientCache"])
                    property_data = next(v['property'] for k, v in data.items() if 'ForSale' in k or 'ForRent' in k)
                except KeyError:
                    try:
                        property_data = data["props"]["pageProps"]["initialData"]['building']
                        return (property_data, [])
                    except KeyError:
                        property_data = data["props"]["initialReduxState"]["gdp"]["building"]
                        return (property_data, [])

                return (property_data, [])
            else:
                # other times it's in Apollo cache
                data = sel.css('script#hdpApolloPreloadedData::text').get()
                data = json.loads(json.loads(data)['apiCache'])
                property_data = next(v['property'] for k, v in data.items() if 'ForSale' in k or 'ForRent' in k)
                return (property_data, [])
        except Exception as e:
            log.warning(f"failure getting {url}: {e}")
            return (None, [])
    urls = [url for url in urls if not '/apartments/' in url]

    results = []
    raw_results = await gather_with_concurrency(2, *[scrape(url) for url in urls])
    return [raw_result[0] for raw_result in raw_results]
    more_to_scrape = []
    for r in raw_results:
        if r[1]:
            for item in r[1]:
                more_to_scrape.append(item)
            continue
        results.append(r[0])
    remaining_results = await gather_with_concurrency(2, *[scrape("https://zillow.com" + item["hdpUrl"]) for item in more_to_scrape])
    log.info(f"found {len(remaining_results)} nested result")
    return results + [r[0] for r in remaining_results]
import pprint

def print_important_keys(important_keys, p):
    newdict = {}
    for key in important_keys:
        try:
            if type(key) == tuple:
                newdict[key[0] + '_' + key[1]] = p[key[0]][key[1]]
            else:
                newdict[key] = p[key]
        except Exception as e:
            if key not in ['pageViewCount', 'numberOfUnitsTotal']:
                log.warning(e)
            newdict[key] = None

    try:
        return newdict
    except Exception as exc:
        log.warning(exc)
        return None

def populate_db(records, table):
    if not records: return
    log.info(f"Updating db table {table} with {len(records)} records")
    with sqlite3.connect('zillow.db') as conn:
        #query = 'drop TABLE if exists properties'
        #conn.execute(query)
        #conn.commit()
        # Create table with columns from dictionary keys
        keys = records[0].keys()
        keys_create = [k + (" STRING NOT NULL PRIMARY KEY" if k == "zpid" else "") for k in keys]
        query = f'CREATE TABLE IF NOT EXISTS {table} (' + ', '.join(keys_create) + ')'
        conn.execute(query)
        conn.commit()

        # Insert records from dictionaries as rows in table
        for record in records:
            values = [str(record[key]) if isinstance(record[key], Location) else record[key] for key in keys]
            query = f'insert or replace INTO {table} (' + ', '.join(keys) + ') VALUES (' + ', '.join(['?' for i in range(len(keys))]) + ')'
            conn.execute(query, values)

        # Commit changes
        conn.commit()
    log.info(f"Insert/updated {len(records)} records into {table}")

def process_rental_result(p, result):
    if result:
        if result['hdpUrl']:
            result['hdpUrl'] = 'https://www.zillow.com' + result['hdpUrl']
        result['zpid'] = str(result['zpid'])

def process_result(p, result):
    if result:
        try:
            if result['address_zipcode'] not in zip_codes_of_interest.keys() or not result['hdpUrl']:
                return
        except Exception as e:
            log.warning(e)
        result['price_rent_ratio'] = 0
        if 'rentZestimate' in result and result['rentZestimate']:
            result['price_rent_ratio'] = result['price'] / result['rentZestimate']

        result['ppsf'] = 0
        if 'price' in result and 'livingArea' in result and result['livingArea'] and result['livingArea'] > 0:
            result['ppsf'] = result['price'] / result['livingArea']
        if ((' #' in p['address']['streetAddress'] or
            ' apt ' in p['address']['streetAddress'].lower() or
            ' unit ' in p['address']['streetAddress'].lower()
             )
            and result['homeType'] == 'MULTI_FAMILY'):
            result['homeType'] = 'CONDO-PROBABLY'
        result['location'] = zip_codes_of_interest[result['address_zipcode']]
        result['hdpUrl'] = 'https://www.zillow.com' + result['hdpUrl']
        result['rentalIncome'] = None
        if (result['resoFacts_totalActualRent'] and
            result['resoFacts_totalActualRent'] > 0 and
            result['resoFacts_totalActualRent'] != 19998 and # filter out some crap data
            result['rentZestimate'] != None and
            result['resoFacts_totalActualRent'] * 3 > result['rentZestimate']): # make sure the total actual rent is not bonkers
            result['rentalIncome'] = result['resoFacts_totalActualRent']
        else:
            result['rentalIncome'] = result['rentZestimate']
        result['estimatedCapRate'] = None
        if result['rentalIncome']:
            result['estimatedCapRate'] = result['rentalIncome'] * 12 / result['price']

        result['postedDate'] = None
        if 'datePostedString' in p:
            result['postedDate'] = datetime.datetime.strptime(p['datePostedString'], '%Y-%m-%d').date()

        result['soldDate'] = None
        if 'dateSoldString' in p:
            result['soldDate'] = datetime.datetime.strptime(p['dateSoldString'], '%Y-%m-%d').date()
        result['lastListPrice'] = result['price']
        result['originalListPrice'] = result['price']
        result['saleDifference'] = 0
        result['saleFraction'] = 1.0
        result['saleDifferenceOriginal'] = 0
        result['saleFractionOriginal'] = 1.0
        result['bedroomList'] = None
        if result['homeStatus'] == 'FOR_SALE':
            location = zip_codes_of_interest[result['address_zipcode']]
            city = location.city
            tax_rate = city.tax_rate
            exemption = city.exemption
            result['estimatedMonthlyTax'] = (result['price'] / 1000 * tax_rate - exemption) / 12
            result['estimatedMaintenance'] = 0.01 * result['price'] / 12
            m = mortgage.Loan(interest=data.interest_rate, principal=result['price'] * (1 - data.down_payment), term=30)
            result['mortgagePayment'] = int(float(m.schedule(1).payment))
            result['firstPrincipalPayment'] = int(float(m.schedule(1).principal))

            bedrooms = result['bedrooms']
            units = result['numberOfUnitsTotal']

            if units and bedrooms:
                unit_list = [bedrooms // units] * units
                remainder = bedrooms % units
                i = 0
                while remainder:
                    unit_list[i] += 1
                    remainder -= 1
                    i += 1
                result['bedroomList'] = ','.join(str(u) for u in unit_list)

        if result['homeStatus'] in ('SOLD', 'RECENTLY_SOLD'):
            if p['priceHistory'][0]['event'] in ('Listing removed', 'Contingent'):
                result['homeStatus'] = 'PENDING' # We don't have good final price info/sale date yet

        if result['homeStatus'] in ('SOLD', 'RECENTLY_SOLD'):
            result['daysOnZillow'] = None # this number seems to be unreliable for sold properties. we try to figure it out below, but we don't always have the data for when properties went on the market
        if result['homeStatus'] in ('SOLD', 'RECENTLY_SOLD'):
            listing_removed_recent = False
            for item in p['priceHistory']:
                if item['event'] == 'Listing removed':
                    listing_removed_date = datetime.datetime.strptime(item['date'], '%Y-%m-%d').date()
                    listing_removed_recent = (result['soldDate'] - listing_removed_date).days < 90
                    break

            for item in p['priceHistory']:
                if item['event'] in ('Contingent', 'Price change', 'Listed for sale') and item['postingIsRental'] is False:
                    result['lastListPrice'] = item['price']
                    result['saleDifference'] = result['price'] - item['price']
                    result['saleFraction'] = result['price'] / item['price']
                    break

            listing_date = None
            for item in p['priceHistory']:
                date = datetime.datetime.strptime(item['date'], '%Y-%m-%d').date()
                if item['event'] in ('Listed for sale') and (result['soldDate'] - date).days < 365 and item['postingIsRental'] is False:
                    result['originalListPrice'] = item['price']
                    result['saleDifferenceOriginal'] = result['price'] - item['price']
                    result['saleFractionOriginal'] = result['price'] / item['price']
                    listing_date = date
                    days_listed_for_sale = (result['soldDate'] - datetime.datetime.strptime(item['date'], '%Y-%m-%d').date()).days
                    #if days_listed_for_sale < 730: # just guessing.. there are some cases where the first/only one is way in the past
                    #    result['daysOnZillow'] = days_listed_for_sale
                    break

            off_the_market_date = result['soldDate']
            for item in p['priceHistory']:
                if item['event'] in ('Contingent', 'Listing removed') and (datetime.datetime.strptime(item['date'], '%Y-%m-%d').date() > listing_date if listing_date else True) and item['postingIsRental'] is False:
                    off_the_market_date =  datetime.datetime.strptime(item['date'], '%Y-%m-%d').date()
                    break

            if listing_date and (off_the_market_date - listing_date).days > 0:
                result['daysOnZillow'] =  (off_the_market_date - listing_date).days
    else:
        log.warning(f"Had trouble with property: {p['address']}")
        return

def chunk_list(lst, chunk_size):
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

async def do_rents(sales, args, func_map):
    important_keys = ['zpid', 'abbreviatedAddress', ('address', 'zipcode'), 'bedrooms', 'bathrooms', 'daysOnZillow', 'livingArea', 'pageViewCount', 'price', 'lastSoldPrice', 'rentZestimate', 'zestimate', 'hdpUrl', 'location', 'latitude', 'longitude']

    more_info_urls = []
    results = []
    for sale in sales:
        if sale.get('isBuilding', False):
            more_info_urls.append(sale['detailUrl'])
        else:
            result = sale['hdpData']['homeInfo']
            result['hdpUrl'] = sale['detailUrl']
            results.append(result)

    abbr_props_outer = []
    for i, p in enumerate(results):
        p["abbreviatedAddress"] = p["streetAddress"]
        p['address'] = {'zipcode': p["zipcode"]}
        p['pageViewCount'] = None
        p['numberOfUnitsTotal'] = None
        p['lastSoldPrice'] = None
        p["rentZestimate"] = None
        p["zestimate"] = None
        p['location'] = zip_codes_of_interest.get(p['zipcode'], None)
        if not "livingArea" in p:
            p["livingArea"] = None
        result = print_important_keys(important_keys, p)
        if not result['price']:
            continue
        process_rental_result(p, result)
        abbr_props_outer.append(result)
    populate_db(abbr_props_outer, table=func_map[args.mode]['table'])

    urls = more_info_urls
    chunk_size = 10
    for i, chunk in enumerate(chunk_list(urls, chunk_size)):
        results = []
        log.info(f"Processing chunk #{i}/{len(urls)//chunk_size} of urls")
        property_data = await scrape_properties(
                [("https://zillow.com" if not 'https://' in url else "") + url for url in chunk], memoize= not 'detail' in args.refresh
        )
        for x in property_data:
            if x is None: continue
            if x["ungroupedUnits"]:
                for p in x["ungroupedUnits"]:
                    if p["listingType"] != "FOR_RENT":
                        continue
                    result = p
                    p['address'] = {'zipcode': x["zipcode"]}
                    try:
                        result["walkScore"] = None
                        try:
                            result["walkScore"] = x["walkScore"]["walkscore"]
                        except Exception:
                            pass
                        result["bedrooms"] = p["beds"]
                        result["bathrooms"] = p["baths"]
                    except Exception as e:
                        import pdb; pdb.set_trace()
                        pass
                    result["abbreviatedAddress"] = x["address"]["streetAddress"] + " " + (p["unitNumber"] or "")
                    result["daysOnZillow"] = None
                    result["livingArea"] = p["sqft"]
                    result["pageViewCount"] = None
                    result["lastSoldPrice"] = None
                    result["rentZestimate"] = None
                    result["zestimate"] = None
                    result['location'] = zip_codes_of_interest.get(x['zipcode'], None)
                    results.append(result)
            elif x["floorPlans"]:
                for floor_plan in x["floorPlans"]:
                    if floor_plan["units"]:
                        for unit in floor_plan["units"]:
                            result = unit
                            result['address'] = {'zipcode': x["zipcode"]}
                            result["walkScore"] = x["walkScore"]
                            result["bedrooms"] = floor_plan["beds"]
                            result["bathrooms"] = floor_plan["baths"]
                            result["abbreviatedAddress"] = x["address"]["streetAddress"] + " " + (p["unitNumber"] or "")
                            result["daysOnZillow"] = None
                            result["livingArea"] = floor_plan["sqft"]
                            result["pageViewCount"] = None
                            result["lastSoldPrice"] = None
                            result["rentZestimate"] = None
                            result["zestimate"] = None
                            result['location'] = zip_codes_of_interest.get(x['zipcode'], None)
                            result['zpid'] = floor_plan['zpid'] + '-' + (p["unitNumber"] or "")
                            result["hdpUrl"] = x["bdpUrl"]
                            results.append(result)

        for x in results:
            for k in ['ungroupedUnits', 'amenityPhotos', 'assignedSchools', 'big', 'breadcrumbs', 'galleryPhotos', 'galleryAmenityPhotos', 'nearbyBuildingLinks',  'comps', 'attributionInfo', 'adTargets', 'ZoDsFsUpsellTop', 'contactFormRenderData', 'listingMetadata', 'nearbyCities', 'nearbyHomes', 'nearbyNeighborhoods', 'nearbyZipcodes', 'onsiteMessage', 'photos', 'responsivePhotos', 'responsivePhotosOriginalRatio','staticMap', 'topNavJson', 'tourEligibility']:
                if x and k in x:
                    x.pop(k)

        results = [p for p in results if p is not None]
        abbr_props = []
        for i, p in enumerate(results):
            result = print_important_keys(important_keys, p)
            if not result['price']:
                continue
            process_rental_result(p, result)
            abbr_props.append(result)
        populate_db(abbr_props, table=func_map[args.mode]['table'])


async def do_properties(args, urls):
    chunk_size = 10
    for i, chunk in enumerate(chunk_list(urls, chunk_size)):
        results = []
        log.info(f"Processing chunk #{i}/{len(urls)//chunk_size} of urls")
        property_data = await scrape_properties(
                [("https://zillow.com" if not 'https://' in url else "") + url for url in chunk], memoize= not 'detail' in args.refresh
        )
        for x in property_data:
            if x is None:
                continue
            for k in ['comps', 'attributionInfo', 'adTargets', 'ZoDsFsUpsellTop', 'contactFormRenderData', 'listingMetadata', 'nearbyCities', 'nearbyHomes', 'nearbyNeighborhoods', 'nearbyZipcodes', 'onsiteMessage', 'photos', 'responsivePhotos', 'responsivePhotosOriginalRatio','staticMap', 'topNavJson', 'tourEligibility']:
                if x and k in x:
                    x.pop(k)

        important_keys = ['zpid', 'abbreviatedAddress', ('address', 'zipcode'), 'homeType', 'homeStatus', 'bedrooms', 'bathrooms', 'daysOnZillow', 'livingArea', 'lotSize', 'numberOfUnitsTotal', 'pageViewCount', 'price', ('resoFacts', 'totalActualRent'), 'rentZestimate', 'zestimate', 'hdpUrl', 'latitude', 'longitude']

        abbr_props = []
        property_data = [p for p in property_data if p is not None]
        for p in property_data:
            if "ungroupedUnits" in p and p["ungroupedUnits"]:
                for u in p["ungroupedUnits"]:
                    if u["listingType"] != "FOR_SALE":
                        continue
                    more_data = await scrape_properties([("https://zillow.com" if not 'https://' in url else "") + url for url in [u["hdpUrl"]]], memoize= not 'detail' in args.refresh)
                    result = print_important_keys(important_keys, more_data[0])
                    process_result(more_data[0], result)
                    if result['address_zipcode'] in zip_codes_of_interest:
                        abbr_props.append(result)
                continue
            result = print_important_keys(important_keys, p)
            process_result(p, result)
            if result['address_zipcode'] in zip_codes_of_interest:
                abbr_props.append(result)

        abbr_props.sort(key=lambda x: x['price_rent_ratio'])

        log.info(f"Found {len(abbr_props)} properties")

        populate_db(abbr_props, table=func_map[args.mode]['table'])

func_map = {'properties': {'func': search_sale, 'table':'properties'},
            'sold': {'func':search_sold, 'table': 'properties'},
            'rent': {'func':search_rent, 'table': 'rentals'},
}
async def run():
    parser = argparse.ArgumentParser(description="Zillow scraper")
    parser.add_argument("--refresh", nargs='+', help="Specify functions to refresh from scraping ('search', 'detail')", default=[])
    parser.add_argument("--mode", type=str, help="Specify mode ('properties', 'sold', 'rent')", default="properties")
    parser.add_argument("--per-search-number", type=int, help="Number of ZIP codes per search call", default=5)
    parser.add_argument("--zip", nargs='+', default=zip_codes_of_interest)
    args = parser.parse_args()

    search_func = func_map[args.mode]['func']

    #rentals = await search_rent("New Haven, CT", session)
    #pprint.pprint(rentals)
    sales = []
    zips_per_search = args.per_search_number
    zips = args.zip
    for i in range(0, len(zips), zips_per_search):
        cur_zips = list(zips)[i:i+zips_per_search]
        sales += await search_func(cur_zips, memoize=not 'search' in args.refresh)

    for x in sales:
        if args.mode == 'rent' and '/homedetails/' in x['detailUrl']:
            x['detailUrl'] = re.sub('/homedetails/', '/b/', x['detailUrl'])
    urls = []
    for sale in sales:
        try:
            zip_code = None
            if 'addressZipcode' in sale:
                zip_code = sale['addressZipcode']
            elif 'hdpData' in sale:
                zip_code = sale['hdpData']['homeInfo']['zipcode']
        except KeyError:
            import pdb; pdb.set_trace()
        if zip_code in zip_codes_of_interest or zip_code is None:
            urls.append(sale['detailUrl'])
    urls = sorted(urls)
    #pprint.pprint(sales)
    #property_data = await scrape_properties(
    #    ["https://www.zillow.com/b/aalto57-new-york-ny-5twVDd/"], session=session
    #)
    if args.mode == 'rent':
        await do_rents(sales, args, func_map)
    else:
        await do_properties(args, urls)

if __name__ == "__main__":
    asyncio.run(run())
