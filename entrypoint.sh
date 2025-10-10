#!/bin/bash
# entrypoint.sh

if [ "$1" = "server" ]; then
    python fl_server.py
elif [ "$1" = "client" ]; then
    python fl_client.py
else
    echo "Usage: $0 [server|client]"
    exit 1
fi