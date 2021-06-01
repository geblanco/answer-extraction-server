#!/bin/bash

export ANSWER_EXTRACTION_INSTANCES=${ANSWER_EXTRACTION_INSTANCES:-1}

for instance in `seq 1 $ANSWER_EXTRACTION_INSTANCES`; do
  # stop the server
  docker stop "answer_extraction_${instance}"
done

exit 0
