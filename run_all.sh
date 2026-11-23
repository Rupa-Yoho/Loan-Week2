#!/bin/bash
set -e
for t in task1_concepts task2_pipeline task3_models task4_evaluation;do
python $t.py>outputs/$t.log;done
