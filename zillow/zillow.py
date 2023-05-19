import asyncio
import json
import re
from random import randint
from typing import List
from urllib.parse import urlencode, quote
import pprint
import time
import pickle
import os
import sqlite3

from loguru import logger as log
from parsel import Selector
from scrapfly import ScrapeConfig, ScrapflyClient
import requests
from data import zip_codes_of_interest

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
        print("calling Zillow: {args}")
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
            if kwargs.get('memoize', False):
                kwargs.pop('memoize')
                cache_key = (_convert_dict_to_tuple(args), _convert_dict_to_tuple(kwargs))
                if cache_key not in cache:
                    result = await func(*args, **kwargs)
                    cache[cache_key] = pickle.dumps(result)
                    with open(filename, 'wb') as f:
                        pickle.dump(cache, f)
                else:
                    result = pickle.loads(cache[cache_key])
            else:
                result = func(*args, **kwargs)
            return result
        return wrapper
    return memoize

m_requests_get =  memoize('cache.pkl')(rate_limiter(1, 1, 1)(print_args(requests.get)))

async def _search(zips: List[str], session: ScrapflyClient, filters: dict = None, categories=("cat1", "cat2")) -> List[dict]:
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
    filters = {"mapBounds":{"north":42.44534296278287,"east":-70.97269326611328,"south":42.24767622824179,"west":-71.26039773388672},"isMapVisible":False,"filterState":{"sortSelection":{"value":"globalrelevanceex"},"isSingleFamily":{"value":False},"isTownhouse":{"value":False},"isCondo":{"value":False},"isLotLand":{"value":False},"isApartment":{"value":False},"isManufactured":{"value":False},"isApartmentOrCondo":{"value":False},"price":{"max":price_limit}},"isListVisible":True,"mapZoom":12,"category":"cat1","pagination":{}}
    regions = {"regionSelection": [{"regionId": int(z) + 56514} for z in zips]}
    filters.update(regions)
    query_data = filters
    url = "https://www.zillow.com/search/GetSearchPageState.htm?"
    found = []
    # cat1 - Agent Listings
    # cat2 - Other Listings
    for category in categories:
        search_query_stats = json.dumps(query_data)
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
        response = await m_requests_get(url + urlencode(full_query, quote_via=quote), headers=headers, memoize=True)
        data = json.loads(response.content)
        _total = data["categoryTotals"][category]["totalResultCount"]
        if _total > 500:
            log.warning(f"query has more results ({_total}) than 500 result limit ")
        else:
            log.info(f"found {_total} results for zips: {zips}")
        map_results = data[category]["searchResults"]["mapResults"]
        found.extend(map_results)
    return found


async def search_sale(zips: List[str], session: ScrapflyClient) -> List[dict]:
    """search properties that are for sale"""
    return await _search(zips=zips, session=session)


async def search_rent(zips: List[str], session: ScrapflyClient) -> List[dict]:
    """search properites that are for rent"""
    log.info(f"scraping rent search for: {query}")
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
    return await _search(zips=zips, session=session, filters=filters, categories=["cat1"])


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

async def scrape_properties(urls: List[str], session: ScrapflyClient):
    """scrape zillow properties"""

    async def scrape(url):
        #result = await session.async_scrape(
        #    ScrapeConfig(url=url, asp=True, country="US")
        #)
        #response = result.upstream_result_into_response()
        #api_response = response.text
        response = await m_requests_get(url, headers=headers, memoize=True)
        api_response = response.text
        sel = Selector(text=api_response)
        data = sel.css("script#__NEXT_DATA__::text").get()
        try:
            if data:
                # some properties are located in NEXT DATA cache
                data = json.loads(data)
                try:
                    data = json.loads(data["props"]["pageProps"]["gdpClientCache"])
                    property_data = next(v['property'] for k, v in data.items() if 'ForSale' in k)
                except KeyError:
                    property_data = data["props"]["pageProps"]["initialData"]['building']
                return property_data
            else:
                # other times it's in Apollo cache
                data = sel.css('script#hdpApolloPreloadedData::text').get()
                data = json.loads(json.loads(data)['apiCache'])
                property_data = next(v['property'] for k, v in data.items() if 'ForSale' in k)
                return property_data
        except Exception as e:
            import pdb; pdb.set_trace()

    return await gather_with_concurrency(2, *[scrape(url) for url in urls])

import pprint

def print_important_keys(important_keys, property):
    newdict = {}
    try:
        for key in important_keys:
            if type(key) == tuple:
                newdict[key[0] + '_' + key[1]] = property[key[0]][key[1]]
            else:
                newdict[key] = property[key]
    except Exception as e:
        log.warning(e)
        return None

    try:
        newdict['price_rent_ratio'] = 0
        if 'rentZestimate' in newdict and newdict['rentZestimate']:
            newdict['price_rent_ratio'] = newdict['price'] / newdict['rentZestimate']

        newdict['ppsf'] = 0
        if 'price' in newdict and 'livingArea' in newdict and newdict['livingArea'] and newdict['livingArea'] > 0:
            newdict['ppsf'] = newdict['price'] / newdict['livingArea']
        return newdict
    except Exception as exc:
        log.warning(exc)
        return None

def populate_db(records):
    data = records
    with sqlite3.connect('zillow.db') as conn:
        query = 'drop TABLE if exists properties'
        conn.execute(query)
        conn.commit()
        # Create table with columns from dictionary keys
        keys = data[0].keys()
        query = 'CREATE TABLE IF NOT EXISTS properties (' + ', '.join(keys) + ')'
        conn.execute(query)
        conn.commit()

        # Insert data from dictionaries as rows in table
        for record in data:
            values = [record[key] for key in keys]
            query = 'INSERT INTO properties (' + ', '.join(keys) + ') VALUES (' + ', '.join(['?' for i in range(len(keys))]) + ')'
            conn.execute(query, values)

        # Commit changes
        conn.commit()

async def run():
    with ScrapflyClient(key="scp-live-19d8c8607fc549e58baee2b303e453a3", max_concurrency=2) as session:
        #rentals = await search_rent("New Haven, CT", session)
        #pprint.pprint(rentals)
        sales = []
        for i in range(0, len(zip_codes_of_interest), 5):
            zips = list(zip_codes_of_interest)[i:i+5]
            sales += await search_sale(zips, session)

        urls = sorted(list(set([x['detailUrl'] for x in sales])))
        #pprint.pprint(sales)
        #property_data = await scrape_properties(
        #    ["https://www.zillow.com/b/aalto57-new-york-ny-5twVDd/"], session=session
        #)
        property_data = await scrape_properties(
                ["https://zillow.com" + url for url in urls], session=session
        )
        for x in property_data:
            for k in ['comps', 'attributionInfo', 'adTargets', 'ZoDsFsUpsellTop', 'contactFormRenderData', 'listingMetadata', 'nearbyCities', 'nearbyHomes', 'nearbyNeighborhoods', 'nearbyZipcodes', 'onsiteMessage', 'photos', 'responsivePhotos', 'responsivePhotosOriginalRatio','staticMap', 'topNavJson', 'tourEligibility']:
                if k in x:
                    x.pop(k)

        important_keys = ['bedrooms', 'abbreviatedAddress', ('address', 'zipcode'), ('resoFacts', 'totalActualRent'), 'bathrooms', 'daysOnZillow', 'homeType', 'livingArea', 'lotSize', 'numberOfUnitsTotal', 'pageViewCount', 'price', 'rentZestimate', 'zestimate', 'hdpUrl']

        abbr_props = []
        for p in property_data:
            result = print_important_keys(important_keys, p)
            if result:
                if result['address_zipcode'] not in zip_codes_of_interest.keys():
                    continue
                result['location'] = zip_codes_of_interest[result['address_zipcode']]
                result['hdpUrl'] = 'https://www.zillow.com' + result['hdpUrl']
                result['rentalIncome'] = None
                if result['resoFacts_totalActualRent'] and result['resoFacts_totalActualRent'] > 0 and result['resoFacts_totalActualRent'] != 19998:
                    result['rentalIncome'] = result['resoFacts_totalActualRent']
                else:
                    result['rentalIncome'] = result['rentZestimate']
                result['estimatedCapRate'] = None
                if result['rentalIncome']:
                    result['estimatedCapRate'] = result['rentalIncome'] * 12 / result['price']

                abbr_props.append(result)
            else:
                log.warning(f"Had trouble with property: {p['address']}")

        abbr_props.sort(key=lambda x: x['price_rent_ratio'])

        log.info(f"Found {len(abbr_props)} properties")

        populate_db(abbr_props)


if __name__ == "__main__":
    asyncio.run(run())
