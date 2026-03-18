nohup python clustering.py --data 3 > clustering_obj3_all_nomin.log 2>&1 &
nohup python clustering.py --data 2 > clustering_obj2_all_nomin.log 2>&1 &
nohup python clustering.py --data 1 > clustering_obj1_all_nomin.log 2>&1 &

#nohup python clustering.py --data 3 --min_scale 0.5 > clustering_obj3_all.log 2>&1 &
#nohup python clustering.py --data 2 --min_scale 0.5 > clustering_obj2_all.log 2>&1 &
#nohup python clustering.py --data 1 --min_scale 0.5 > clustering_obj1_all.log 2>&1 &