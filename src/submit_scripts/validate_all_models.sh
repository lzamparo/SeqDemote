#! /bin/bash

pushd ~/projects/SeqDemote/src/models/torch_models/embedded_ATAC_models
declare -a models
for file in $(ls -1 *.py | sort -n)
do
    models=("${models[@]}" "$file")
done
echo "Got all models, ${#models[@]}"
popd

pushd ~/projects/SeqDemote/results/BindSpace_embedding_extension
declare -a savestates
for file in $(ls -1 *.ptm | sort -n)
do 
   savestates=("${savestates[@]}" "$file")   
done
echo "Got all save states, ${#savestates[@]}"
popd

model_prefix="torch_models/embedded_ATAC_models"

for (( i=0; i<${#models[@]}; i++ ));
do
	model=${models[$i]}
        savestate=${savestates[$i]}
	#echo "-env all, MODEL=$model_prefix/$model, STATE=$savestate"
	bsub -env "all, MODEL=$model_prefix/$model, STATE=$savestate" < validate_model.lsf
done
