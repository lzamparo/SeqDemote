#! /bin/bash

pushd ~/projects/SeqDemote/src/models/torch_models/embedded_ATAC_models

model_prefix="torch_models/embedded_ATAC_models"
models=$(ls *.py)
popd
for model in $models
do
	#echo "-env all, MODEL=$model_prefix/$model < blah blah"
	bsub -env "all, MODEL=$model_prefix/$model" < hyperparameter_search.lsf
done
