import argparse
import json

from milkplan import report


def main():
    p = argparse.ArgumentParser(prog="milkplan")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="validate, backtest, forecast, allocate and write outputs")
    r.add_argument("--geo", default="DK")
    r.add_argument("--offline", action="store_true", help="use the committed Eurostat snapshot")
    a = p.parse_args()
    res = report.run(geo=a.geo, offline=a.offline)
    print(json.dumps({k: res[k] for k in ("best_model", "mean_mape_best", "mean_mape_seasonal_naive",
                                          "interval_coverage_80", "spot_kt_by_scenario")}, indent=2))


if __name__ == "__main__":
    main()
