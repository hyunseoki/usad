#/usr/bin/bash


# for train_site in 'b c' 'a b' 'a c'
# do
#     for val_site in a b c
#     do
#         if [[ ! "$train_site" =~ "$val_site" ]]; then
#             python main.py --train_site $train_site \
#                                  --val_site "$val_site" \
#                                  --comments train_${train_site// /_}_val_${val_site}
#         fi
#     done
# done


for train_site in 'b c' 'a b' 'a c'
do
    for val_site in a b c
    do
        python main.py --train_site $train_site \
                       --val_site "$val_site" \
                       --comments train_${train_site// /_}_val_${val_site}
    done
done