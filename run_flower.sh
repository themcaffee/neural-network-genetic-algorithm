#!/bin/bash

celery -A main flower --broker=amqp://myuser:mypassword@localhost:5672/myvhost