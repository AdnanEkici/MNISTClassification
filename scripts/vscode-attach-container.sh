#!/usr/bin/env bash
CONTAINER_NAME="/$1"
hex=$(printf \{\"containerName\"\:\""$CONTAINER_NAME"\"\} | od -A n -t x1 | tr -d '[\n\t ]')
code --folder-uri vscode-remote://attached-container+${hex}/$2
