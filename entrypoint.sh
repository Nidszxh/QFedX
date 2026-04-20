#!/bin/bash
# entrypoint.sh

export PYTHONPATH=/workspace/src:$PYTHONPATH

if [ "$1" = "server" ]; then
    cd /workspace/src
    python fl_server.py
elif [ "$1" = "client" ]; then
    cd /workspace/src
    python fl_client.py
elif [ "$1" = "run" ]; then
    cd /workspace/src
    python run.py "${@:2}"
else
    exec "$@"
fi
