for i in `seq 2014 2018`; do 
    train=''
    for j in `seq $i 2018`; do 
        train="$train $j"
    done
    python -m genolearn.train O157-region-subset/$i/rf RandomForestClassifier data_config.json rf_config.json -train $train -test 2019 -K 100 1000 10000 100000 1000000 -order fisher-scores.npz -order_key $i -target_subset "Asia" "C. America" "C. Europe" "M. East" "N. Africa" "S. Europe" "Subsaharan Africa" "UK"
    python -m genolearn.train O157-region-subset/$i/lr LogisticRegression     data_config.json lr_config.json -train $train -test 2019 -K 100 1000 10000 100000         -order fisher-scores.npz -order_key $i -target_subset "Asia" "C. America" "C. Europe" "M. East" "N. Africa" "S. Europe" "Subsaharan Africa" "UK"
done