# Gunicorn reads this file automatically from the working directory, so these
# settings apply whether Render starts the app from the Procfile or from a start
# command in its dashboard. Flags on the command line override anything here, so
# the start command should be just `gunicorn app:app`.

# One process. Every process holds its own copy of both models (~350 MB with the
# libraries loaded), and two of them outgrow a 512 MB instance -- which is what
# "Worker was sent SIGKILL! Perhaps out of memory?" was reporting.
workers = 1

# Threads instead of processes for concurrency. Requests run on the thread pool
# while the worker's main loop keeps checking in with the arbiter, so one slow
# page can no longer get the whole worker killed as timed out.
worker_class = "gthread"
threads = 4

timeout = 120
graceful_timeout = 30

# No max_requests: recycling the worker threw away the loaded models and made
# the replacement re-download and refit everything, and nothing here leaks.
