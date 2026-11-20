"""Placeholder: writes both tables empty with the right schema. Replaced by the real reference pipeline."""
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

out = Path(os.environ["OUTPUT_DIR"])
U = pa.schema([("user_id", pa.string()), ("day", pa.date32()), ("sessions", pa.int64()), ("events", pa.int64()),
               ("revenue_cents", pa.int64()), ("top_category", pa.string())])
C = pa.schema([("campaign_id", pa.string()), ("channel", pa.string()), ("day", pa.date32()), ("sessions", pa.int64()),
               ("purchasing_sessions", pa.int64()), ("revenue_cents", pa.int64()), ("assisted_sessions", pa.int64())])
for name, schema in (("user_daily", U), ("campaign_daily", C)):
    (out / name).mkdir(parents=True, exist_ok=True)
    pq.write_table(schema.empty_table(), out / name / "part-0.parquet")
