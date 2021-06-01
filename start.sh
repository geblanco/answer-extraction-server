#!/bin/bash

# default to one instance
export ANSWER_EXTRACTION_INSTANCES=${ANSWER_EXTRACTION_INSTANCES:-1}

./stop.sh

for instance in `seq 1 $ANSWER_EXTRACTION_INSTANCES`; do
  docker rm "answer_extraction_${instance}"
  host_port=$(( 9019 + $instance ))
  docker run -d -p $host_port:8000 -v /home/gblanco/models:/usr/app/models --name "answer_extraction_${instance}" answer_extraction:latest --route answer_extraction --model models/multibert/cased_v100
done

exit 0

