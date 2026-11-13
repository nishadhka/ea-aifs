"""source.coop I/O for the Colab hindcast (COLAB_PLAN.md).

Reads are anonymous (the e4drr-project repository is public). Writes use the short-lived
S3 credentials of the `source-coop` CLI, which hc_colab.sh pipes over ssh into
/root/.sc.env (mode 600) and refreshes before they expire. The file is re-read on every
write, so a refresh takes effect without restarting anything. Credentials never appear in
`colab exec` code, logs or this repository.

Layout (one date MMDD, year YYYY, member NNN):
    forecasts/fp16FahamuAIFSv2/hc-inputs-tmp/<MMDD>/<YYYY>/input_state_member_<NNN>.pkl
    forecasts/fp16FahamuAIFSv2/hc-products/<MMDD>/<YYYY>/m<NNN>/...   (+ DONE, written last)
"""

import os
import time

import boto3
from boto3.s3.transfer import TransferConfig
from botocore import UNSIGNED
from botocore.config import Config

BUCKET = "e4drr-project"
ENDPOINT = "https://data.source.coop"
BASE = "forecasts/fp16FahamuAIFSv2"
CRED_FILE = os.environ.get("SC_CRED_FILE", "/root/.sc.env")
XFER = TransferConfig(multipart_threshold=64 << 20, multipart_chunksize=64 << 20,
                      max_concurrency=16)


def input_key(mmdd, year, member):
    return f"{BASE}/hc-inputs-tmp/{mmdd}/{year}/input_state_member_{member:03d}.pkl"


def product_prefix(mmdd, year, member):
    return f"{BASE}/hc-products/{mmdd}/{year}/m{member:03d}/"


def _anon():
    return boto3.client("s3", endpoint_url=ENDPOINT, region_name="us-east-1",
                        config=Config(signature_version=UNSIGNED,
                                      retries={"max_attempts": 10, "mode": "adaptive"}))


def _authed():
    env = {}
    with open(CRED_FILE) as fh:
        for line in fh:
            line = line.strip().removeprefix("export ")
            if "=" in line:
                k, v = line.split("=", 1)
                env[k] = v.strip().strip('"').strip("'")
    return boto3.client("s3", endpoint_url=ENDPOINT, region_name="us-east-1",
                        aws_access_key_id=env["AWS_ACCESS_KEY_ID"],
                        aws_secret_access_key=env["AWS_SECRET_ACCESS_KEY"],
                        aws_session_token=env.get("AWS_SESSION_TOKEN"),
                        config=Config(retries={"max_attempts": 10, "mode": "adaptive"}))


def size(key):
    """Object size in bytes, or None if absent."""
    try:
        return _anon().head_object(Bucket=BUCKET, Key=key)["ContentLength"]
    except Exception:
        return None


def ls(prefix):
    out = []
    for pg in _anon().get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix):
        out += [(o["Key"], o["Size"]) for o in pg.get("Contents", [])]
    return out


def get(key, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    _anon().download_file(BUCKET, key, path, Config=XFER)
    return path


def put(path, key, tries=5):
    """Upload and read the size back. Retries the whole upload: source.coop answers the
    odd 520 mid-multipart (2026-11-12, 1 of 20 pkls)."""
    for i in range(tries):
        try:
            _authed().upload_file(path, BUCKET, key, Config=XFER)
            got = size(key)
            if got != os.path.getsize(path):
                raise RuntimeError(f"read-back of {key}: {got} bytes, expected {os.path.getsize(path)}")
            return key
        except Exception as e:
            if i == tries - 1:
                raise
            print(f"  put {key}: {type(e).__name__} {str(e)[:120]}; retry {i+1}", flush=True)
            time.sleep(15 * (i + 1))


def put_bytes(data, key):
    _authed().put_object(Bucket=BUCKET, Key=key, Body=data)
    return key


def rm(key):
    _authed().delete_object(Bucket=BUCKET, Key=key)
