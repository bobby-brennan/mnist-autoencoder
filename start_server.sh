#!/bin/bash
FLASK_APP=server.py flask run --host 0.0.0.0 --port 3000 & export TO_KILL=$!
