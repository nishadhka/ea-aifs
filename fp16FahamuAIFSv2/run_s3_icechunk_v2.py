#!/usr/bin/env python3
"""
AIFS-ENS-2.0 FP16 inference -> Icechunk store on **source.coop S3**.

S3 counterpart of ``run_local_icechunk_v2.py``. The rollout, the ``(member, time,
values)`` schema from ``ICECHUNK_PATH_A.md`` and the one-commit-per-member ACID model
are identical -- this script only swaps the storage backend and the input source, then
delegates to ``run_local_icechunk_v2.run(..., repo_factory=...)`` so there is exactly
one copy of the inference loop.

    input   gs://ea_aifs_w1/<date>/input_v2_pre50r1/input_state_member_NNN.pkl
            (read with coiled-data-e4drr_202505.json)
    output  s3://e4drr-project/forecasts/fp16FahamuAIFSv2_n320/<YYYYMMDD>/
            (endpoint https://data.source.coop, path-style addressing)

Nothing but the ~1 GB input pkl of the member being run touches local disk; every
forecast byte goes straight from the GPU into the remote store, one commit per step.

Credentials
-----------
Writing needs the 12 h source.coop STS token from
``run-pre50r1-dates/SOURCE_COOP_CREDENTIALS.md`` (reading a published store does not --
see its §0). They are taken from, in order:

1. the ``source-coop`` CLI credential cache, and
2. ``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY`` / ``AWS_SESSION_TOKEN`` in the
   environment or in ``--env-file`` (default ``.env`` beside this script).

The cache is preferred *for refresh*: the repo is opened with
``s3_refreshable_credentials``, so when the token expires icechunk calls back and the
callback re-reads the cache. A run longer than the token's 12 h therefore survives a
``source-coop login`` in another shell instead of dying mid-rollout -- which matters
because a 50-member 960 h ensemble takes longer than one token lives.

``--require-hours`` refuses to start when the credential cannot outlive the estimated
run, the same guard ``mirror_all_to_source_coop.sh`` applies before each store: better
to fail in the first second than unattended at hour nine.

Usage::

    source .env                       # or rely on the CLI cache alone
    # smoke test: one member, 72 h, into a throwaway prefix
    python run_s3_icechunk_v2.py --date 20260212_0000 --members 1 --lead-time 72 \
        --store-prefix forecasts/fp16FahamuAIFSv2_n320/_smoke --n-members 1

    # the real cycle: 50 members, 40 days, resumable
    python run_s3_icechunk_v2.py --date 20260212_0000 --members 1-50 --lead-time 960 \
        --skip-existing --cleanup-pkl
"""

import os
import sys
import json
import datetime
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import run_local_icechunk_v2 as local_run

# --- this cycle's fixed coordinates ---------------------------------------
# Input lives in a different bucket/prefix from the standard v2 pipeline: the
# pre-Cy50r1 inputs are built by run-pre50r1-dates/ecmwf_opendata_pkl_input_aifsens_v2_pre50r1.py.
DEFAULT_GCS_BUCKET = "ea_aifs_w1"
DEFAULT_GCS_SUBPATH = "input_v2_pre50r1"
DEFAULT_SERVICE_ACCOUNT = "coiled-data-e4drr_202505.json"

DEFAULT_STORE_BUCKET = "e4drr-project"
DEFAULT_STORE_ROOT = "forecasts/fp16FahamuAIFSv2_n320"
SOURCE_COOP_ENDPOINT = "https://data.source.coop"
SOURCE_COOP_REGION = "us-east-1"

# The source-coop CLI writes its STS token here (see SOURCE_COOP_CREDENTIALS.md §3).
# $SOURCE_COOP_CACHE overrides, and the two defaults cover a coiled image and a laptop.
CRED_CACHE_CANDIDATES = (
    os.environ.get("SOURCE_COOP_CACHE", ""),
    "/opt/coiled/cache/source-coop/credentials/_default.json",
    os.path.expanduser("~/.cache/source-coop/credentials/_default.json"),
)

# Measured on an L4 at FP16 + 16 chunks (LOAD_TEST_RESULTS.md §3): ~4.4 s per 6 h step.
# Used only to estimate whether the credential outlives the run.
SECONDS_PER_STEP = 4.6


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------
def read_env_file(path):
    """Parse ``export K=V`` / ``K=V`` lines from a .env without executing it.

    Sourcing it would be simpler but this script is also imported, and a .env that
    holds Ceph keys alongside the source.coop ones should not be run as shell.
    """
    out = {}
    if not path or not os.path.exists(path):
        return out
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def cred_cache_path():
    for p in CRED_CACHE_CANDIDATES:
        if p and os.path.exists(p):
            return p
    return None


def credentials_from_cache():
    """(dict, expiry) from the source-coop CLI cache, or (None, None)."""
    p = cred_cache_path()
    if not p:
        return None, None
    try:
        with open(p) as fh:
            d = json.load(fh)
        # The cache file is snake_case; `source-coop creds --format credential-process`
        # prints the same fields PascalCase. Accept either, so a cache written by a
        # future CLI version (or a hand-placed credential_process blob) still works.
        def pick(*names):
            for n in names:
                if n in d:
                    return d[n]
            return None
        creds = {"AWS_ACCESS_KEY_ID": pick("access_key_id", "AccessKeyId"),
                 "AWS_SECRET_ACCESS_KEY": pick("secret_access_key", "SecretAccessKey"),
                 "AWS_SESSION_TOKEN": pick("session_token", "SessionToken")}
        if not creds["AWS_ACCESS_KEY_ID"] or not creds["AWS_SECRET_ACCESS_KEY"]:
            raise KeyError("no access key in cache")
        exp = pick("expiration", "Expiration")
        expiry = (datetime.datetime.fromisoformat(exp.replace("Z", "+00:00"))
                  if exp else None)
        return creds, expiry
    except Exception as e:
        print(f"[creds] cache at {p} unreadable ({e}); falling back to the environment")
        return None, None


def credentials_from_env(env_file):
    """(dict, None) from the process env, with --env-file as the fallback source.

    No expiry: the three AWS_* variables carry none, which is exactly why
    SOURCE_COOP_CREDENTIALS.md §5 says to verify the grant rather than assume it.
    """
    merged = dict(read_env_file(env_file))
    merged.update({k: v for k, v in os.environ.items() if k.startswith("AWS_")})
    if not merged.get("AWS_ACCESS_KEY_ID") or not merged.get("AWS_SECRET_ACCESS_KEY"):
        return None, None
    return {"AWS_ACCESS_KEY_ID": merged["AWS_ACCESS_KEY_ID"],
            "AWS_SECRET_ACCESS_KEY": merged["AWS_SECRET_ACCESS_KEY"],
            "AWS_SESSION_TOKEN": merged.get("AWS_SESSION_TOKEN")}, None


def resolve_credentials(env_file):
    """(creds, expiry, source) -- cache first so the refresh callback can re-read it."""
    creds, expiry = credentials_from_cache()
    if creds:
        return creds, expiry, f"source-coop CLI cache ({cred_cache_path()})"
    creds, expiry = credentials_from_env(env_file)
    if creds:
        return creds, expiry, f"environment / {env_file}"
    return None, None, None


def hours_left(expiry):
    if expiry is None:
        return None
    return (expiry - datetime.datetime.now(datetime.timezone.utc)).total_seconds() / 3600


class SourceCoopCredentials:
    """Callable icechunk invokes whenever the current token has expired.

    Re-reads the cache on each call, so a `source-coop login` in another shell refreshes
    a running job. Falls back to the env when only a .env exists.

    A class rather than a closure because icechunk **pickles** the callback
    (``s3_refreshable_credentials`` -> ``pickle.dumps(get_credentials)``), and a local
    function is not picklable. Only ``env_file`` is pickled -- no secret is serialised.
    """

    def __init__(self, env_file):
        self.env_file = env_file

    def __call__(self):
        import icechunk
        creds, expiry, _ = resolve_credentials(self.env_file)
        if not creds:
            raise RuntimeError(
                "source.coop credentials unavailable at refresh time. Run "
                "`source-coop login --duration 12h` (see SOURCE_COOP_CREDENTIALS.md §3).")
        left = hours_left(expiry)
        print("[creds] refreshed"
              + (f", {left:.2f} h remaining" if left is not None else ""), flush=True)
        return icechunk.S3StaticCredentials(
            access_key_id=creds["AWS_ACCESS_KEY_ID"],
            secret_access_key=creds["AWS_SECRET_ACCESS_KEY"],
            session_token=creds.get("AWS_SESSION_TOKEN"),
            expires_after=expiry)


# ---------------------------------------------------------------------------
# Storage backend
# ---------------------------------------------------------------------------
def split_s3_uri(uri):
    """'s3://bucket/prefix' -> ('bucket', 'prefix')."""
    if not uri.startswith("s3://"):
        raise ValueError(f"expected an s3:// URI, got {uri!r}")
    rest = uri[len("s3://"):].strip("/")
    bucket, _, prefix = rest.partition("/")
    if not bucket or not prefix:
        raise ValueError(f"{uri!r} needs both a bucket and a prefix")
    return bucket, prefix


def make_repo_factory(env_file):
    """Factory passed to run_local_icechunk_v2.run() -- opens an S3-backed repo.

    force_path_style is required: data.source.coop serves s3://bucket/key as
    /bucket/key, not bucket.data.source.coop (same flag SOURCE_COOP_CREDENTIALS.md §0
    uses for the anonymous read path).
    """
    import icechunk

    get_credentials = SourceCoopCredentials(env_file)

    def _open(uri):
        bucket, prefix = split_s3_uri(uri)
        storage = icechunk.s3_storage(
            bucket=bucket,
            prefix=prefix,
            endpoint_url=SOURCE_COOP_ENDPOINT,
            region=SOURCE_COOP_REGION,
            force_path_style=True,
            from_env=False,
            get_credentials=get_credentials,
        )
        return icechunk.Repository.open_or_create(storage)

    return _open


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="AIFS-ENS-2.0 FP16 inference into an Icechunk store on source.coop S3")
    ap.add_argument("--date", required=True, help="cycle date YYYYMMDD_0000")
    ap.add_argument("--members", default="1", help="'1-50' or '1,2,3' or '7'")
    ap.add_argument("--lead-time", type=int, default=local_run.DEFAULT_LEAD_TIME,
                    help="forecast lead time (hours), default 960")
    ap.add_argument("--input-dir", default=None,
                    help="local staging dir for the pkls (default ./input_states_<date>)")

    # --- output store ---
    ap.add_argument("--store-bucket", default=DEFAULT_STORE_BUCKET)
    ap.add_argument("--store-prefix", default=None,
                    help=f"default '{DEFAULT_STORE_ROOT}/<YYYYMMDD>'")
    ap.add_argument("--env-file", default=os.path.join(HERE, ".env"),
                    help="file holding the source.coop AWS_* vars (default .env here)")
    ap.add_argument("--require-hours", type=float, default=None,
                    help="refuse to start unless the credential has at least this many "
                         "hours left (default: the estimated run time)")
    ap.add_argument("--force", action="store_true",
                    help="start even if the credential is estimated to expire mid-run "
                         "(the run is resumable with --skip-existing)")

    # --- input source ---
    ap.add_argument("--bucket", default=DEFAULT_GCS_BUCKET, help="GCS input bucket")
    ap.add_argument("--gcs-input-prefix", default=None,
                    help=f"default '<date>/{DEFAULT_GCS_SUBPATH}'")
    ap.add_argument("--service-account", default=DEFAULT_SERVICE_ACCOUNT)

    # --- passthrough to the shared runner ---
    ap.add_argument("--n-members", type=int, default=local_run.DEFAULT_N_MEMBERS)
    ap.add_argument("--precision", default=local_run.INFERENCE_PRECISION,
                    choices=["16", "32"])
    ap.add_argument("--chunks", type=int, default=local_run.INFERENCE_NUM_CHUNKS)
    ap.add_argument("--float-size", default="f4", choices=["f4", "f2"])
    ap.add_argument("--commit-every", type=int, default=1)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--write-hours", default="all")
    ap.add_argument("--no-tag", action="store_true")
    ap.add_argument("--cleanup-pkl", action="store_true",
                    help="delete each member's pkl after it is written (a 50-member "
                         "run stages ~46 GB otherwise)")
    ap.add_argument("--grid", default="n320", choices=["n320", "o96"])
    ap.add_argument("--native-store", default=None,
                    help="second store for --native-vars, as 's3://bucket/prefix'")
    ap.add_argument("--native-vars", default="msl,tp,2t")
    ap.add_argument("--write-threads", type=int, default=16,
                    help="concurrent chunk PUTs per step. Each step is ~120 "
                         "independent 2.1 MiB uploads and a single stream only reaches "
                         "~1.7 MiB/s to data.source.coop; 8 threads reach ~12.9 and 24 "
                         "~16.5 (default 16)")
    ap.add_argument("--write-retries", type=int, default=6,
                    help="attempts per chunk write / commit. data.source.coop returns "
                         "an intermittent empty-body error that icechunk does not "
                         "retry itself, which would otherwise abort a member "
                         "(default 6, exponential backoff)")
    args = ap.parse_args()

    datestr = args.date.split("_")[0]
    store_prefix = args.store_prefix or f"{DEFAULT_STORE_ROOT}/{datestr}"
    store_uri = f"s3://{args.store_bucket}/{store_prefix.strip('/')}"
    gcs_prefix = args.gcs_input_prefix or f"{args.date}/{DEFAULT_GCS_SUBPATH}"
    input_dir = args.input_dir or os.path.join(HERE, f"input_states_{datestr}")

    service_account = args.service_account
    if not os.path.isabs(service_account):
        service_account = os.path.join(HERE, service_account)
    if not os.path.exists(service_account):
        print(f"ERROR: GCS service account not found: {service_account}")
        return 1

    # --- credential preflight (SOURCE_COOP_CREDENTIALS.md §5) --------------
    creds, expiry, source = resolve_credentials(args.env_file)
    if not creds:
        print("ERROR: no source.coop credentials found.\n"
              "       Run: source-coop login --duration 12h   (then `source .env`)\n"
              f"       Looked in the CLI cache and {args.env_file}")
        return 1
    if not creds.get("AWS_SESSION_TOKEN"):
        print("WARNING: no AWS_SESSION_TOKEN — source.coop STS credentials need one.")

    members = local_run.parse_member_range(args.members)
    n_steps = args.lead_time // local_run.TIME_STEP_HOURS
    est_h = len(members) * n_steps * SECONDS_PER_STEP / 3600
    left = hours_left(expiry)

    print("=" * 70)
    print("AIFS-ENS-2.0 FP16  ->  Icechunk on source.coop S3")
    print("=" * 70)
    print(f"Input:       gs://{args.bucket}/{gcs_prefix}/  ({os.path.basename(service_account)})")
    print(f"Store:       {store_uri}  ({SOURCE_COOP_ENDPOINT})")
    print(f"Credentials: {source}"
          + (f" | {left:.2f} h left (expires {expiry:%Y-%m-%d %H:%M} UTC)"
             if left is not None else " | expiry unknown (env vars carry none)"))
    print(f"Upload:      {args.write_threads} concurrent chunk PUTs | "
          f"{args.write_retries} attempts each")
    print(f"Estimated:   ~{est_h:.1f} h of inference "
          f"({len(members)} members x {n_steps} steps x {SECONDS_PER_STEP}s)")
    print("=" * 70)

    need = args.require_hours if args.require_hours is not None else est_h
    if left is not None and left < need:
        msg = (f"credential has {left:.2f} h left but the run needs ~{need:.1f} h")
        if not args.force:
            print(f"\nERROR: {msg}.\n"
                  "       Refresh first: source-coop login --duration 12h\n"
                  "       Or pass --force (the run resumes with --skip-existing), or\n"
                  "       --require-hours to lower the bar.")
            return 1
        print(f"\nWARNING: {msg} — continuing because --force was given. "
              "Re-run with --skip-existing after refreshing to finish.")

    ok = local_run.run(
        date_str=args.date, members=members, input_dir=input_dir,
        store_path=store_uri, lead_time=args.lead_time,
        n_members=args.n_members, precision=args.precision,
        num_chunks=args.chunks, float_size=args.float_size,
        tag=not args.no_tag, commit_every=args.commit_every,
        skip_existing=args.skip_existing, write_hours=args.write_hours,
        gcs_fetch=True, bucket=args.bucket, gcs_prefix=gcs_prefix,
        service_account_key=service_account,
        cleanup_pkl=args.cleanup_pkl, grid=args.grid,
        native_store=args.native_store,
        native_vars=[v.strip() for v in args.native_vars.split(",") if v.strip()],
        repo_factory=make_repo_factory(args.env_file),
        write_retries=args.write_retries, write_threads=args.write_threads,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
