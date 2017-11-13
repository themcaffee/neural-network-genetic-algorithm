#!/bin/bash

celery -A main worker --loglevel=info --concurrency=1