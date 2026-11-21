import glob, os, sys
import duckdb, polars as pl, pyarrow.parquet as pq
from pyspark.sql import SparkSession

COLS = {
    "user_daily": ["user_id", "day", "sessions", "events", "revenue_cents", "top_category"],
    "campaign_daily": ["campaign_id", "channel", "day", "sessions", "purchasing_sessions", "revenue_cents", "assisted_sessions"],
}
spark = SparkSession.builder.master("local[2]").config("spark.ui.enabled", "false").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")


def files(d, t):
    p = os.path.join(d, t)
    if os.path.isdir(p):
        return sorted(f for f in glob.glob(os.path.join(p, "**", "*.parquet"), recursive=True)
                      if not os.path.basename(f).startswith((".", "_")))
    return [p + ".parquet"]


def norm(rows):
    return sorted(tuple("" if v is None else str(v) for v in r) for r in rows)


bad = 0
first = {}
for d in sys.argv[1:]:
    for t, cols in COLS.items():
        fs = files(d, t)
        pf = pq.ParquetFile(fs[0])
        print(f"-- {d} {t}: {len(fs)} files, created_by {pf.metadata.created_by!r}, row groups {pf.metadata.num_row_groups}")
        print(pf.schema)
        got = {}
        tb = pq.read_table(fs, columns=cols)
        assert str(tb.schema.field("day").type) == "date32[day]", tb.schema
        got["pyarrow"] = norm(zip(*[tb.column(c).to_pylist() for c in cols]))
        rel = duckdb.sql("select " + ",".join(cols) + " from read_parquet(" + repr(fs) + ")")
        assert str(rel.types[cols.index("day")]) == "DATE", rel.types
        got["duckdb"] = norm(rel.fetchall())
        df = pl.read_parquet(fs).select(cols)
        assert df.schema["day"] == pl.Date, df.schema
        got["polars"] = norm(df.rows())
        for vec in ("true", "false"):
            spark.conf.set("spark.sql.parquet.enableVectorizedReader", vec)
            sd = spark.read.parquet(*["file://" + f for f in fs]).select(*cols)
            assert sd.schema["day"].dataType.typeName() == "date", sd.schema
            if vec == "true":
                print("spark schema:", sd.schema.simpleString())
            got["spark-vec" if vec == "true" else "spark-row"] = norm(tuple(r) for r in sd.collect())
        ref = got["pyarrow"]
        for k, v in got.items():
            if v != ref:
                bad += 1
                diff = sorted(set(v) ^ set(ref))[:5]
                print(f"MISMATCH {t} {k} vs pyarrow: {len(v)} vs {len(ref)} rows, sample {diff}")
        if t in first:
            if ref != first[t]:
                bad += 1
                print(f"MISMATCH {t}: {d} rows differ from {sys.argv[1]}")
        else:
            first[t] = ref
        print(f"{t}: {len(ref)} rows; readers {sorted(got)}")
print("PQCHECK PASS" if bad == 0 else f"PQCHECK FAIL ({bad})")
sys.exit(1 if bad else 0)
