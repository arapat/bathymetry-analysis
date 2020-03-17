dirname=/Users/igpp-jalafate/workbox/bathymetry-analysis/logs/by-cruises/cross-regions/runtime_scores
files=$(ls $dirname)

for filename in $(ls $dirname)
do
    base=${filename%.*}
    model_name=$base.cali.joblib
    proba_name=$base.proba.pkl
    printf '.'
    python calibration.py train --scores $dirname/$filename --model $model_name --result $proba_name
    python calibration.py test --scores $dirname/$filename --model $model_name --result $proba_name
done
echo

