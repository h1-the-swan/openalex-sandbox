# -*- coding: utf-8 -*-

DESCRIPTION = """Get recently published works from UW, filter for collaborations with other institutions, save data to CSV files.
Example usage: python uw_collabs_update.py 2023-03-01 output/ example@uw.edu
"""

import sys, os, time
import requests
from typing import List, Iterable
from pathlib import Path
from datetime import datetime
from timeit import default_timer as timer

try:
    from humanfriendly import format_timespan
except ImportError:

    def format_timespan(seconds):
        return "{:.2f} seconds".format(seconds)


import pandas as pd
import numpy as np


import logging

root_logger = logging.getLogger()
logger = root_logger.getChild(__name__)


def query_publications_data(
    institution_id: str, from_publication_date: str, email: str
) -> List:
    # specify endpoint
    endpoint = "works"

    # build the 'filter' parameter
    filters = ",".join(
        (
            f"institutions.ror:{institution_id}",
            f"from_publication_date:{from_publication_date}",
        )
    )

    # put the URL together
    filtered_works_url = f"https://api.openalex.org/{endpoint}?filter={filters}"
    if email:
        filtered_works_url += f"&mailto={email}"
    logger.debug(f"complete URL with filters:\n{filtered_works_url}")

    cursor = "*"

    select = ",".join(
        (
            "id",
            "ids",
            "title",
            "display_name",
            "publication_year",
            "publication_date",
            "primary_location",
            "open_access",
            "authorships",
            "cited_by_count",
            "is_retracted",
            "is_paratext",
            "updated_date",
            "created_date",
        )
    )

    # loop through pages
    works = []
    loop_index = 0
    logger.info(f"querying works for papers published since {from_publication_date}...")
    while cursor:
        # set cursor value and request page from OpenAlex
        url = f"{filtered_works_url}&select={select}&cursor={cursor}"
        page_with_results = requests.get(url).json()

        results = page_with_results["results"]
        works.extend(results)

        # update cursor to meta.next_cursor
        cursor = page_with_results["meta"]["next_cursor"]
        loop_index += 1
        if loop_index in [5, 10, 20, 50, 100] or loop_index % 500 == 0:
            logger.info(f"{loop_index} api requests made so far")
    logger.info(f"done. made {loop_index} api requests. collected {len(works)} works")
    return works


def get_publications_dataframe(works: Iterable) -> pd.DataFrame:
    data = []
    for work in works:
        for authorship in work["authorships"]:
            if authorship:
                author = authorship["author"]
                author_id = author["id"] if author else None
                author_name = author["display_name"] if author else None
                author_position = authorship["author_position"]
                author_orcid = author.get("orcid", None) if author else None
                for institution in authorship["institutions"]:
                    if institution:
                        institution_id = institution["id"]
                        institution_name = institution["display_name"]
                        institution_country_code = institution["country_code"]
                        institution_lineage = institution.get("lineage", [])
                        data.append(
                            {
                                "work_id": work["id"],
                                "work_title": work["title"],
                                "work_display_name": work["display_name"],
                                "work_publication_year": work["publication_year"],
                                "work_publication_date": work["publication_date"],
                                "author_id": author_id,
                                "author_name": author_name,
                                "author_position": author_position,
                                "author_orcid": author_orcid,
                                "institution_id": institution_id,
                                "institution_name": institution_name,
                                "institution_country_code": institution_country_code,
                                "institution_toplevel": institution_lineage[-1] if institution_lineage else None,
                            }
                        )
    df = pd.DataFrame(data)
    return df


def outside_uw_collab(institution_ids: List[str]) -> bool:
    # Function that takes institution IDs (grouped by works)
    # and returns True if the work has at least one non-UW affiliation
    if all(institution_ids == "https://openalex.org/I201448701"):
        return False
    else:
        return True


def query_institutions_data(institution_ids: Iterable[str], email: str) -> List:
    endpoint = "institutions"
    size = 50
    loop_index = 0
    institutions = []
    logger.info("querying institutions...")
    for list_index in range(0, len(institution_ids), size):
        subset = institution_ids[list_index : list_index + size]
        pipe_separated_ids = "|".join(subset)
        r = requests.get(
            f"https://api.openalex.org/{endpoint}?filter=openalex:{pipe_separated_ids}&per-page={size}&mailto={email}"
        )
        results = r.json()["results"]
        institutions.extend(results)
        loop_index += 1
    logger.info(
        f"collected {len(institutions)} institutions using {loop_index} api calls"
    )
    return institutions


def get_institutions_dataframe(institutions: Iterable) -> pd.DataFrame:
    data = []
    for institution in institutions:
        data.append(
            {
                "id": institution["id"],
                "ror": institution["ror"],
                "display_name": institution["display_name"],
                "country_code": institution["country_code"],
                "type": institution["type"],
                "lineage": institution.get("lineage"),
                "latitude": institution["geo"]["latitude"],
                "longitude": institution["geo"]["longitude"],
                "city": institution["geo"]["city"],
                "region": institution["geo"]["region"],
                "country": institution["geo"]["country"],
                "image_url": institution["image_url"],
                "image_thumbnail_url": institution["image_thumbnail_url"],
            }
        )

    df_institutions = pd.DataFrame(data)
    return df_institutions


def main(args):
    outdir = Path(args.outdir)
    if not outdir.is_dir():
        logger.info(f"Output directory does not exist. Creating directory: {outdir}")
        outdir.mkdir(parents=True)

    uw_id = "https://ror.org/00cvxb145"
    from_publication_date = args.from_publication_date
    email = args.email

    works = query_publications_data(
        institution_id=uw_id, from_publication_date=from_publication_date, email=email
    )

    df_publications = get_publications_dataframe(works)

    # transform() will return a series the same length as the dataframe,
    # which we can store as a column in the dataframe
    df_publications["is_outside_uw_collab"] = df_publications.groupby("work_id")[
        "institution_id"
    ].transform(outside_uw_collab)

    df_collab = df_publications[df_publications["is_outside_uw_collab"]].drop(
        columns="is_outside_uw_collab"
    )
    logger.info(
        f"dataframe has {len(df_collab):,} rows, with {df_collab['work_id'].nunique():,} unique publications"
    )

    institution_ids = pd.concat([df_collab["institution_id"], df_collab['institution_toplevel']]).dropna().unique()
    institutions = query_institutions_data(institution_ids=institution_ids, email=email)
    df_institutions = get_institutions_dataframe(institutions)
    outfp = outdir.joinpath('uw_collab_update_institutions.csv')
    logger.info(f"Saving institutions to: {outfp}")
    df_institutions.to_csv(outfp, index=False)

    df_collab['institution_toplevel_name'] = df_collab['institution_toplevel'].map(df_institutions.set_index('id')['display_name'])
    outfp = outdir.joinpath('uw_collab_update_publications.csv')
    logger.info(f"Saving publications to: {outfp}")
    df_collab.to_csv(outfp, index=False)


if __name__ == "__main__":
    total_start = timer()
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(name)s.%(lineno)d %(levelname)s : %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
    logger.info("{:%Y-%m-%d %H:%M:%S}".format(datetime.now()))
    import argparse

    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("from_publication_date", help="Date, formatted as `yyyy-mm-dd`")
    parser.add_argument(
        "outdir", help="output directory (will be created if it doesn't exist)."
    )
    parser.add_argument(
        "email", help="email address used to identify user to the OpenAlex API"
    )
    parser.add_argument("--debug", action="store_true", help="output debugging info")
    global args
    args = parser.parse_args()
    if args.debug:
        root_logger.setLevel(logging.DEBUG)
        logger.debug("debug mode is on")
    logger.debug(" ".join(sys.argv))
    logger.debug("pid: {}".format(os.getpid()))
    main(args)
    total_end = timer()
    logger.info(
        "all finished. total time: {}".format(format_timespan(total_end - total_start))
    )
