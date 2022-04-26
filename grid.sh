for i in `seq 2014 2018`; do 
    train=''
    for j in `seq $i 2018`; do 
        train="$train $j"
    done
    python -m genolearn.train O157-region/$i/rf RandomForestClassifier data_config.json rf_config.json -train $train -test 2019 -K 100 1000 10000 100000 1000000 -order fisher-score.npz -order_key $i -min_count 15
    python -m genolearn.train O157-region/$i/lr LogisticRegression     data_config.json lr_config.json -train $train -test 2019 -K 100 1000 10000 100000         -order fisher-score.npz -order_key $i -min_count 15
done

