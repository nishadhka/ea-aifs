#!/usr/bin/env python3
"""
Icechunk ensemble output writer for AIFS-ENS-2.0 (Path A — **local** variant).

Implements the writer specified in ``ICECHUNK_PATH_A.md`` §2.1, but targets a
**local filesystem** Icechunk store (``icechunk.local_filesystem_storage``) instead
of GCS. Everything else — the ``(member, time, values)`` schema, the ``(1, n_steps,
n_values)`` chunking that lets members own disjoint chunks, the one-commit-per-member
ACID model — is identical, so this store is a drop-in for the GCS one and reads back
the same way with ``xarray.open_zarr``.

Use for on-box runs where the GPU inference writes straight into a store on a local
disk (e.g. ``/tank/projects/<date>/icechunk_v2``) — no GRIB staging, no upload.

Schema (native N320 reduced-Gaussian cells, no regrid at write time)::

    coords: latitude(values) f4, longitude(values) f4,
            time(time) i4 (seconds since reference_date), member(member) i4
    data:   one (member, time, values) f4/f2 array per model output field
            (ALL fields kept, incl. cos_mwd/sin_mwd — a gain over the GRIB path)

Typical use (single process, sequential members)::

    repo = open_repo_local("/tank/projects/20260625_0000/icechunk_v2")
    init_schema(repo, n_members=50, n_steps=12, n_values=542080,
                var_names=first_state_field_names,
                latitudes=lats, longitudes=lons,
                ref_date=ref_date, timestep_s=6*3600)      # once, before any writer
    w = IcechunkMemberWriter(repo, member_index=member-1)  # member 1 -> index 0
    for state in runner.run(...):
        w.write_step(state)
    w.commit(member)
"""

import numpy as np
import zarr
import icechunk


# ---------------------------------------------------------------------------
# Repository handles
# ---------------------------------------------------------------------------
def open_repo_local(path):
    """Open (or create) a local-filesystem Icechunk repository at ``path``.

    ``path`` is a directory on a local disk; it is created if missing.
    """
    storage = icechunk.local_filesystem_storage(str(path))
    return icechunk.Repository.open_or_create(storage)


def open_repo_gcs(bucket, prefix, service_account_file):
    """Open (or create) a GCS-backed Icechunk repository (parity with Path A doc)."""
    storage = icechunk.gcs_storage(bucket=bucket, prefix=prefix,
                                   service_account_file=service_account_file)
    return icechunk.Repository.open_or_create(storage)


# ---------------------------------------------------------------------------
# Schema (create once, before any member is written)
# ---------------------------------------------------------------------------
def init_schema(repo, n_members, n_steps, n_values, var_names,
                latitudes, longitudes, ref_date, timestep_s,
                float_size="f4", commit_msg=None):
    """Create + commit the (member, time, values) schema on branch ``main``.

    Must run once, committed BEFORE any :class:`IcechunkMemberWriter` opens a
    session. ``var_names`` are the model output field names (derive them from the
    first yielded state so the set always matches the checkpoint exactly).
    ``latitudes``/``longitudes`` are the native N320 cell coordinates (length
    ``n_values``). Returns the commit snapshot id.
    """
    s = repo.writable_session("main")
    root = zarr.group(s.store, overwrite=True)

    root.create_array("latitude", shape=(n_values,), chunks=(n_values,),
                      dtype="f4", dimension_names=("values",))[:] = np.asarray(latitudes, "f4")
    root.create_array("longitude", shape=(n_values,), chunks=(n_values,),
                      dtype="f4", dimension_names=("values",))[:] = np.asarray(longitudes, "f4")

    t = root.create_array("time", shape=(n_steps,), chunks=(n_steps,),
                          dtype="i4", dimension_names=("time",))
    t[:] = (np.arange(1, n_steps + 1) * int(timestep_s)).astype("i4")
    t.attrs["units"] = f"seconds since {ref_date}"
    t.attrs["calendar"] = "gregorian"

    root.create_array("member", shape=(n_members,), chunks=(n_members,),
                      dtype="i4", dimension_names=("member",))[:] = \
        np.arange(1, n_members + 1, dtype="i4")

    for v in var_names:
        root.create_array(v, shape=(n_members, n_steps, n_values),
                          chunks=(1, n_steps, n_values), dtype=float_size,
                          dimension_names=("member", "time", "values"))

    msg = commit_msg or (f"init schema: {n_members}x{n_steps}x{n_values}, "
                         f"{len(var_names)} vars ({float_size})")
    return s.commit(msg)


def schema_exists(repo, sentinel="time"):
    """True if the schema group is already present on ``main`` (idempotent init)."""
    try:
        root = zarr.open_group(repo.readonly_session("main").store, mode="r")
        return sentinel in root
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Per-member writer (one ACID commit per member)
# ---------------------------------------------------------------------------
class IcechunkMemberWriter:
    """Writes ONE member's rollout into the (member, time, values) arrays, then commits.

    ``member_index`` is 0-based (ensemble member N -> index N-1). Because each
    member owns whole ``(1, n_steps, n_values)`` chunks, sequential members never
    touch the same chunk, so per-member commits are conflict-free.
    """

    def __init__(self, repo, member_index, branch="main"):
        self.session = repo.writable_session(branch)
        self.root = zarr.open_group(self.session.store, mode="r+")
        self.m = int(member_index)
        self.n = 0                       # step counter within this member
        self._var_names = None

    def write_step(self, state):
        """Write one yielded inference state (all fields) at the current time index."""
        fields = state["fields"]
        if self._var_names is None:
            self._var_names = set(self.root.array_keys())
        for name, value in fields.items():
            if name in self._var_names:
                self.root[name][self.m, self.n, :] = np.asarray(value).reshape(-1)
        self.n += 1

    def commit(self, member_number):
        """Seal this member as one ACID snapshot; returns the snapshot id."""
        return self.session.commit(f"member {member_number:03d} ({self.n} steps)")
