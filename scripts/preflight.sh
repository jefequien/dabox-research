#!/bin/bash

set -xe

ruff check --fix .
ruff format .
mypy .
