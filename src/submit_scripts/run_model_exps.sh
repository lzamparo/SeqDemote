#! /bin/bash

pushd ~/projects/SeqDemote/src/models/torch_models
models=$(ls *.py)
popd
for model in $models
do
	bsub -env "all, MODEL=$model" < run_model.lsf
done
