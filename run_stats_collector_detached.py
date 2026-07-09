import contextlib
import pathlib
import sys
import time
import traceback


BASE = pathlib.Path(__file__).resolve().parent
STATUS_DIR = BASE / "run_status"
STATUS_DIR.mkdir(exist_ok=True)


def write_status(name, text):
    (STATUS_DIR / name).write_text(str(text))


def run_quietly(label, fn):
    write_status(f"{label}.started", time.strftime("%Y-%m-%d %H:%M:%S"))
    try:
        with open("/dev/null", "w") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                fn()
    except Exception:
        write_status(f"{label}.error", traceback.format_exc())
        write_status(f"{label}.exit", 1)
        raise
    else:
        write_status(f"{label}.done", time.strftime("%Y-%m-%d %H:%M:%S"))
        write_status(f"{label}.exit", 0)


def main():
    import stats_collector

    write_status("stats_collector_runner.pid", str(os.getpid()))
    run_quietly("collect_stats", stats_collector.collect_stats)
    run_quietly("run_noise_sweeps", stats_collector.run_noise_sweeps)


if __name__ == "__main__":
    import os

    try:
        main()
    except Exception:
        sys.exit(1)
