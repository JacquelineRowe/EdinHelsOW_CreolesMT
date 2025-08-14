# filepaths to set 
home_dir="/home/jrowe"
git_repo_path="${home_dir}/creolesMT"
kreyol_mt_data_path="${home_dir}/kreyol-mt-data"
additional_kreyol_mt_data_path="${git_repo_path}/data/additional_kreyolmt_pap_data"

# other filepaths
datadir="${git_repo_path}/data/preprocessed_lusophone"
additional_data_path="${git_repo_path}/new_data"
dataset="set_6"

SPLIT_VAL_TEST=$1 # Bool: set to true to add 1000 of our data each to kmt test and val sets 

if [[ $SPLIT_VAL_TEST == True ]]; then
    dataset="${dataset}_split"
fi
cd $datadir
rm -r $dataset

mkdir $dataset
# create data storage

# this script combines our data with kreyol-mt train and val sets

### concatenate our training data 
## cri 
cat ${additional_data_path}/cri/monolingual/watchtower_cri.syn_eng.txt \
    > ${datadir}/$dataset/ours_all.eng-cri.eng

cat ${additional_data_path}/cri/monolingual/watchtower_cri.txt \
    > ${datadir}/$dataset/ours_all.eng-cri.cri

## kea 
cat ${additional_data_path}/kea/monolingual/kea_blog_by_sentence_filtered.syn_eng.txt \
    ${additional_data_path}/kea/monolingual/kea_textbook_filtered.syn_eng.txt \
    ${additional_data_path}/kea/monolingual/all_lyrics.sys_eng.txt \
    ${additional_data_path}/kea/parallel/watchtower/watchtower_en.txt \
    > ${datadir}/$dataset/ours_all.eng-kea.eng

cat ${additional_data_path}/kea/monolingual/kea_blog_by_sentence_filtered.txt \
    ${additional_data_path}/kea/monolingual/kea_textbook_filtered.txt \
    ${additional_data_path}/kea/monolingual/all_lyrics.kea.txt \
    ${additional_data_path}/kea/parallel/watchtower/watchtower_kea.txt \
    > ${datadir}/$dataset/ours_all.eng-kea.kea

## pap
cat ${additional_data_path}/pap/parallel/bible/bible_eng.txt \
    ${additional_data_path}/pap/parallel/sayings/phrases_en.txt \
    ${additional_data_path}/pap/parallel/watchtower/corsou_dialect/watchtower_en.txt \
    ${additional_data_path}/pap/monolingual/all_lyrics.sys_eng.txt \
    > ${datadir}/$dataset/ours_all.eng-pap.eng

cat ${additional_data_path}/pap/parallel/bible/bible_pap.txt \
    ${additional_data_path}/pap/parallel/sayings/phrases_pap.txt \
    ${additional_data_path}/pap/parallel/watchtower/corsou_dialect/watchtower_pap.txt \
    ${additional_data_path}/pap/monolingual/all_lyrics.pap.txt \
    > ${datadir}/$dataset/ours_all.eng-pap.pap

## pov
cat ${additional_data_path}/pov/monolingual/documentary_transcriptions_by_sentence_filtered.syn_eng.txt \
    ${additional_data_path}/pov/monolingual/monolingual_dictionary_filtered.syn_eng.txt \
    ${additional_data_path}/pov/parallel/article/article_en.txt \
    ${additional_data_path}/pov/parallel/bible/bible_en.txt \
    ${additional_data_path}/pov/parallel/dictionary_sentences/dict_eng.txt \
    ${additional_data_path}/pov/parallel/watchtower/watchtower_en.txt \
    > ${datadir}/$dataset/ours_all.eng-pov.eng

cat ${additional_data_path}/pov/monolingual/documentary_transcriptions_by_sentence_filtered.txt \
    ${additional_data_path}/pov/monolingual/monolingual_dictionary_filtered.txt \
    ${additional_data_path}/pov/parallel/article/article_pov.txt \
    ${additional_data_path}/pov/parallel/bible/bible_pov.txt \
    ${additional_data_path}/pov/parallel/dictionary_sentences/dict_pov.txt \
    ${additional_data_path}/pov/parallel/watchtower/watchtower_pov.txt \
    > ${datadir}/$dataset/ours_all.eng-pov.pov

#aoa
cat ${additional_data_path}/aoa/eng-aoa.eng \
    > ${datadir}/$dataset/ours_all.eng-aoa.eng

cat ${additional_data_path}/aoa/eng-aoa.aoa \
    > ${datadir}/$dataset/ours_all.eng-aoa.aoa

# fab
cat ${additional_data_path}/fab/fab-eng.eng \
    > ${datadir}/$dataset/ours_all.eng-fab.eng

cat ${additional_data_path}/fab/fab-eng.fab \
    > ${datadir}/$dataset/ours_all.eng-fab.fab

#pre
cat ${additional_data_path}/pre/parallel/pre_sentences.eng_syn.txt \
    > ${datadir}/$dataset/ours_all.eng-pre.eng

cat ${additional_data_path}/pre/parallel/pre_sentences.pre.txt \
    > ${datadir}/$dataset/ours_all.eng-pre.pre

## copy kreyol-mt train, test and val data for all langs 
for tgt in pov kea cri pre aoa fab pap
do
src=eng
# train
cat ${kreyol_mt_data_path}/train.${src}-${tgt}.${src} \
    > ${datadir}/$dataset/train_kmt.${src}-${tgt}.${src}

cat ${kreyol_mt_data_path}/train.${src}-${tgt}.${tgt} \
    > ${datadir}/$dataset/train_kmt.${src}-${tgt}.${tgt}

# val
cat ${kreyol_mt_data_path}/validation.${src}-${tgt}.${src} \
    > ${datadir}/$dataset/validation_kmt.${src}-${tgt}.${src}

# copy tgt (creole language) val data
cat ${kreyol_mt_data_path}/validation.${src}-${tgt}.${tgt} \
    > ${datadir}/$dataset/validation_kmt.${src}-${tgt}.${tgt}

# copy src (european language) test data
cat ${kreyol_mt_data_path}/test.${src}-${tgt}.${src} \
    > ${datadir}/$dataset/test_kmt.${src}-${tgt}.${src}

# copy tgt (creole language) test data
cat ${kreyol_mt_data_path}/test.${src}-${tgt}.${tgt} \
    > ${datadir}/$dataset/test_kmt.${src}-${tgt}.${tgt}
done

# add extra papiamento training data from organisers 
tgt=pap
src=eng
cat ${datadir}/$dataset/train_kmt.${src}-${tgt}.${src} \
    ${additional_kreyol_mt_data_path}/train.cleaned.${src} \
    > ${datadir}/$dataset/train_kmt.${src}-${tgt}.${src}

cat ${datadir}/$dataset/train_kmt.${src}-${tgt}.${tgt} \
    ${additional_kreyol_mt_data_path}/train.cleaned.${tgt} \
    > ${datadir}/$dataset/train_kmt.${src}-${tgt}.${tgt}


# make sure any new training data we introduce does not overlap with kreyolmt test sets 
for tgt in pov kea cri pre aoa fab pap
# for tgt in aoa
do
echo "Number of our $tgt sentences removed because they are in kreyol-mt test set:"
src=eng
    python ${git_repo_path}/scripts/data/clean_novel_data.py \
        --path1 $dataset/ours_all.${src}-${tgt}.${src} \
        --path2 $dataset/ours_all.${src}-${tgt}.${tgt} \
        --path3 ${datadir}/$dataset/test_kmt.${src}-${tgt}.${src} \
        --path4 ${datadir}/$dataset/test_kmt.${src}-${tgt}.${tgt} 
done


for tgt in pov kea pap cri pre fab aoa 
do
    if [[ $SPLIT_VAL_TEST == True ]]; then
    # split out 1,000 of OUR kea, pov, pap and cri train data for test and 10% for validation 
        src=eng
            echo $tgt
            python ${git_repo_path}/scripts/data/split_val_test_data.py \
                --path1 $dataset/ours_all.${src}-${tgt}.${src} \
                --path2 $dataset/ours_all.${src}-${tgt}.${tgt}

        # copy our val data after split 
        cat ${datadir}/${dataset}/ours_all.${src}-${tgt}.${src}_1000val \
            > ${datadir}/$dataset/validation_ours.${src}-${tgt}.${src}

        cat ${datadir}/${dataset}/ours_all.${src}-${tgt}.${tgt}_1000val \
            > ${datadir}/$dataset/validation_ours.${src}-${tgt}.${tgt}

        # # make sure there are no sentences that were previously used for kreyol-mt training in our test set 
        python ${git_repo_path}/scripts/data/clean_novel_data.py \
                --path1 $dataset/ours_all.${src}-${tgt}.${src}_1000test \
                --path2 $dataset/ours_all.${src}-${tgt}.${tgt}_1000test \
                --path3 ${datadir}/$dataset/train_kmt.${src}-${tgt}.${src} \
                --path4 ${datadir}/$dataset/train_kmt.${src}-${tgt}.${tgt} \
            
        # copy our test data after split 
        cat ${datadir}/${dataset}/ours_all.${src}-${tgt}.${src}_1000test \
            > ${datadir}/$dataset/test_ours.${src}-${tgt}.${src}

        cat ${datadir}/${dataset}/ours_all.${src}-${tgt}.${tgt}_1000test \
            > ${datadir}/$dataset/test_ours.${src}-${tgt}.${tgt}

        # val
        cat ${datadir}/${dataset}/validation_ours.${src}-${tgt}.${src} \
            ${datadir}/${dataset}/validation_kmt.${src}-${tgt}.${src} \
            > ${datadir}/$dataset/validation.${src}-${tgt}.${src}

        cat ${datadir}/${dataset}/validation_ours.${src}-${tgt}.${tgt} \
            ${datadir}/${dataset}/validation_kmt.${src}-${tgt}.${tgt} \
            > ${datadir}/$dataset/validation.${src}-${tgt}.${tgt}

        # test 
        cat ${datadir}/${dataset}/test_ours.${src}-${tgt}.${src} \
            ${datadir}/${dataset}/test_kmt.${src}-${tgt}.${src} \
            > ${datadir}/$dataset/test.${src}-${tgt}.${src}

        cat ${datadir}/${dataset}/test_ours.${src}-${tgt}.${tgt} \
            ${datadir}/${dataset}/test_kmt.${src}-${tgt}.${tgt} \
            > ${datadir}/$dataset/test.${src}-${tgt}.${tgt}
    else
    # otherwise clean just save kmt val and test sets as main test sets 
        cat ${datadir}/${dataset}/validation_kmt.${src}-${tgt}.${src} \
            > ${datadir}/$dataset/validation.${src}-${tgt}.${src}

        cat ${datadir}/${dataset}/validation_kmt.${src}-${tgt}.${tgt} \
            > ${datadir}/$dataset/validation.${src}-${tgt}.${tgt}

        cat ${datadir}/${dataset}/test_kmt.${src}-${tgt}.${src} \
            > ${datadir}/$dataset/test.${src}-${tgt}.${src}
        
        cat ${datadir}/${dataset}/test_kmt.${src}-${tgt}.${tgt} \
            > ${datadir}/$dataset/test.${src}-${tgt}.${tgt} 

    fi
done

for tgt in pov kea pap cri pre fab aoa
do
    # copy our training data (with val/test removed if setting true)
    cat ${datadir}/${dataset}/ours_all.${src}-${tgt}.${src} \
        > ${datadir}/$dataset/train_ours.${src}-${tgt}.${src}

    cat ${datadir}/${dataset}/ours_all.${src}-${tgt}.${tgt} \
        > ${datadir}/$dataset/train_ours.${src}-${tgt}.${tgt}
    # combine with kmt
    cat ${datadir}/${dataset}/train_ours.${src}-${tgt}.${src} \
        ${datadir}/${dataset}/train_kmt.${src}-${tgt}.${src} \
        > ${datadir}/$dataset/train.${src}-${tgt}.${src}

    cat ${datadir}/${dataset}/train_ours.${src}-${tgt}.${tgt} \
        ${datadir}/${dataset}/train_kmt.${src}-${tgt}.${tgt} \
        > ${datadir}/$dataset/train.${src}-${tgt}.${tgt}

done

# remove temp files now renamed properly 
rm ${datadir}/${dataset}/ours_all.*

# # clean training data and replace characters for eng-creoles
for tgt in pov pap kea cri aoa fab pre 
do
    # filter and clean our training data and combined training data
    python ${git_repo_path}/scripts/data/clean_data.py \
        --path1 $dataset/train.${src}-${tgt}.${src} \
        --path2 $dataset/train.${src}-${tgt}.${tgt}
    files=("${datadir}/$dataset/train.eng-${tgt}.${tgt}_filtered" \
            "${datadir}/$dataset/train.eng-${tgt}.eng_filtered")
    for f in "${files[@]}"; 
    do
        echo $f
        sed -i "s/\“/@\"/g" $f
        sed -i "s/\”/調\"/g" $f
        sed -i "s/\“/付\"/g" $f
        sed -i "s/\’/혼\'/g" $f
        sed -i "s/\‘/ච\'/g" $f
        sed -i "s/\—/완\-/g" $f
        sed -i "s/\–/罪\-/g" $f
        sed -i "s/\«/\<\</g" $f
        sed -i "s/\»/\>\>/g" $f
        sed -i "s/\‚/\,/g" $f
    done

    python ${git_repo_path}/scripts/data/clean_data.py \
        --path1 $dataset/train_ours.${src}-${tgt}.${src} \
        --path2 $dataset/train_ours.${src}-${tgt}.${tgt}
    files=("${datadir}/$dataset/train_ours.eng-${tgt}.${tgt}_filtered" \
            "${datadir}/$dataset/train_ours.eng-${tgt}.eng_filtered")
    for f in "${files[@]}"; 
    do
        echo $f
        sed -i "s/\“/@\"/g" $f
        sed -i "s/\”/調\"/g" $f
        sed -i "s/\“/付\"/g" $f
        sed -i "s/\’/혼\'/g" $f
        sed -i "s/\‘/ච\'/g" $f
        sed -i "s/\—/완\-/g" $f
        sed -i "s/\–/罪\-/g" $f
        sed -i "s/\«/\<\</g" $f
        sed -i "s/\»/\>\>/g" $f
        sed -i "s/\‚/\,/g" $f
    done

    # filter and clean kmt train, val and test sets 
    python ${git_repo_path}/scripts/data/clean_data.py \
        --path1 $dataset/train_kmt.${src}-${tgt}.${src} \
        --path2 $dataset/train_kmt.${src}-${tgt}.${tgt}
    files=("${datadir}/$dataset/train_kmt.eng-${tgt}.${tgt}_filtered" \
            "${datadir}/$dataset/train_kmt.eng-${tgt}.eng_filtered")
    for f in "${files[@]}"; 
    do
        echo $f
        sed -i "s/\“/@\"/g" $f
        sed -i "s/\”/調\"/g" $f
        sed -i "s/\“/付\"/g" $f
        sed -i "s/\’/혼\'/g" $f
        sed -i "s/\‘/ච\'/g" $f
        sed -i "s/\—/완\-/g" $f
        sed -i "s/\–/罪\-/g" $f
        sed -i "s/\«/\<\</g" $f
        sed -i "s/\»/\>\>/g" $f
        sed -i "s/\‚/\,/g" $f
    done

    python ${git_repo_path}/scripts/data/clean_test_data.py \
        --path1 $dataset/validation_kmt.${src}-${tgt}.${src} \
        --path2 $dataset/validation_kmt.${src}-${tgt}.${tgt}
    files=("${datadir}/$dataset/validation_kmt.eng-${tgt}.${tgt}_filtered" \
            "${datadir}/$dataset/validation_kmt.eng-${tgt}.eng_filtered")
    for f in "${files[@]}"; 
    do
        echo $f
        sed -i "s/\“/@\"/g" $f
        sed -i "s/\”/調\"/g" $f
        sed -i "s/\“/付\"/g" $f
        sed -i "s/\’/혼\'/g" $f
        sed -i "s/\‘/ච\'/g" $f
        sed -i "s/\—/완\-/g" $f
        sed -i "s/\–/罪\-/g" $f
        sed -i "s/\«/\<\</g" $f
        sed -i "s/\»/\>\>/g" $f
        sed -i "s/\‚/\,/g" $f
    done

    python ${git_repo_path}/scripts/data/clean_test_data.py \
        --path1 $dataset/test_kmt.${src}-${tgt}.${src} \
        --path2 $dataset/test_kmt.${src}-${tgt}.${tgt}
    files=("${datadir}/$dataset/test_kmt.eng-${tgt}.${tgt}_filtered" \
            "${datadir}/$dataset/test_kmt.eng-${tgt}.eng_filtered")
    for f in "${files[@]}"; 
    do
        echo $f
        sed -i "s/\“/@\"/g" $f
        sed -i "s/\”/調\"/g" $f
        sed -i "s/\“/付\"/g" $f
        sed -i "s/\’/혼\'/g" $f
        sed -i "s/\‘/ච\'/g" $f
        sed -i "s/\—/완\-/g" $f
        sed -i "s/\–/罪\-/g" $f
        sed -i "s/\«/\<\</g" $f
        sed -i "s/\»/\>\>/g" $f
        sed -i "s/\‚/\,/g" $f
    done

    if [[ $SPLIT_VAL_TEST == True ]]; then
        # if we've added our own val and test data in, clean the full sets: 
        python ${git_repo_path}/scripts/data/clean_test_data.py \
            --path1 $dataset/validation.${src}-${tgt}.${src} \
            --path2 $dataset/validation.${src}-${tgt}.${tgt}
        files=("${datadir}/$dataset/validation.eng-${tgt}.${tgt}_filtered" \
                "${datadir}/$dataset/validation.eng-${tgt}.eng_filtered")
        for f in "${files[@]}"; 
        do
            echo $f
            sed -i "s/\“/@\"/g" $f
            sed -i "s/\”/調\"/g" $f
            sed -i "s/\“/付\"/g" $f
            sed -i "s/\’/혼\'/g" $f
            sed -i "s/\‘/ච\'/g" $f
            sed -i "s/\—/완\-/g" $f
            sed -i "s/\–/罪\-/g" $f
            sed -i "s/\«/\<\</g" $f
            sed -i "s/\»/\>\>/g" $f
            sed -i "s/\‚/\,/g" $f
        done

        python ${git_repo_path}/scripts/data/clean_test_data.py \
            --path1 $dataset/test.${src}-${tgt}.${src} \
            --path2 $dataset/test.${src}-${tgt}.${tgt}
        files=("${datadir}/$dataset/test.eng-${tgt}.${tgt}_filtered" \
                "${datadir}/$dataset/test.eng-${tgt}.eng_filtered")
        for f in "${files[@]}"; 
        do
            echo $f
            sed -i "s/\“/@\"/g" $f
            sed -i "s/\”/調\"/g" $f
            sed -i "s/\“/付\"/g" $f
            sed -i "s/\’/혼\'/g" $f
            sed -i "s/\‘/ච\'/g" $f
            sed -i "s/\—/완\-/g" $f
            sed -i "s/\–/罪\-/g" $f
            sed -i "s/\«/\<\</g" $f
            sed -i "s/\»/\>\>/g" $f
            sed -i "s/\‚/\,/g" $f
        done

        python ${git_repo_path}/scripts/data/clean_test_data.py \
            --path1 $dataset/validation_ours.${src}-${tgt}.${src} \
            --path2 $dataset/validation_ours.${src}-${tgt}.${tgt}
        files=("${datadir}/$dataset/validation_ours.eng-${tgt}.${tgt}_filtered" \
                "${datadir}/$dataset/validation_ours.eng-${tgt}.eng_filtered")
        for f in "${files[@]}"; 
        do
            echo $f
            sed -i "s/\“/@\"/g" $f
            sed -i "s/\”/調\"/g" $f
            sed -i "s/\“/付\"/g" $f
            sed -i "s/\’/혼\'/g" $f
            sed -i "s/\‘/ච\'/g" $f
            sed -i "s/\—/완\-/g" $f
            sed -i "s/\–/罪\-/g" $f
            sed -i "s/\«/\<\</g" $f
            sed -i "s/\»/\>\>/g" $f
            sed -i "s/\‚/\,/g" $f
        done
        python ${git_repo_path}/scripts/data/clean_test_data.py \
            --path1 $dataset/test_ours.${src}-${tgt}.${src} \
            --path2 $dataset/test_ours.${src}-${tgt}.${tgt}
        files=("${datadir}/$dataset/test_ours.eng-${tgt}.${tgt}_filtered" \
                "${datadir}/$dataset/test_ours.eng-${tgt}.eng_filtered")
        for f in "${files[@]}"; 
        do
            echo $f
            sed -i "s/\“/@\"/g" $f
            sed -i "s/\”/調\"/g" $f
            sed -i "s/\“/付\"/g" $f
            sed -i "s/\’/혼\'/g" $f
            sed -i "s/\‘/ච\'/g" $f
            sed -i "s/\—/완\-/g" $f
            sed -i "s/\–/罪\-/g" $f
            sed -i "s/\«/\<\</g" $f
            sed -i "s/\»/\>\>/g" $f
            sed -i "s/\‚/\,/g" $f
        done
    fi
done 

# add pov_norm, syn cri and portuguese data into the mix 
# por 
cat ${additional_data_path}/por/train.por-eng.eng \
    > ${datadir}/$dataset/train.eng-por.eng

cat ${additional_data_path}/por/train.por-eng.por \
    > ${datadir}/$dataset/train.eng-por.por
# filter all and kmt train data 

python ${git_repo_path}/scripts/data/clean_data.py \
    --path1 ${datadir}/$dataset/train.eng-por.eng \
    --path2 ${datadir}/$dataset/train.eng-por.por
files=("${datadir}/$dataset/train.eng-por.eng_filtered" \
        "${datadir}/$dataset/train.eng-por.por_filtered")
for f in "${files[@]}"; 
do
    echo $f
    sed -i "s/\“/@\"/g" $f
    sed -i "s/\”/調\"/g" $f
    sed -i "s/\“/付\"/g" $f
    sed -i "s/\’/혼\'/g" $f
    sed -i "s/\‘/ච\'/g" $f
    sed -i "s/\—/완\-/g" $f
    sed -i "s/\–/罪\-/g" $f
    sed -i "s/\«/\<\</g" $f
    sed -i "s/\»/\>\>/g" $f
    sed -i "s/\‚/\,/g" $f
done
    
# pov 
cat ${additional_data_path}/pov/parallel/test_kmt.eng-pov.pov_filtered_norm \
    > ${datadir}/$dataset/test_kmt.eng-pov.pov_filtered_norm

cat ${additional_data_path}/pov/parallel/test.eng-pov.pov_filtered_norm \
    > ${datadir}/$dataset/test.eng-pov.pov_filtered_norm

# sample from eng-syncri data (eng)
for NUM_K_SAMPLES in 5 25 100
do 
    LABEL="${NUM_K_SAMPLES}k"
    NUM_SAMPLES=$((NUM_K_SAMPLES * 1000))

    ENG_FILE="${additional_data_path}/por/train.por-eng.eng"
    CRI_FILE="${additional_data_path}/cri/parallel/train.por-eng.cri"
    TOTAL_LINES=$(wc -l < "$ENG_FILE")
    shuf -i 1-"$TOTAL_LINES" -n "$NUM_SAMPLES" | sort -n > cri_tmp.indexes
    awk 'NR==FNR{idx[$1]; next} FNR in idx' cri_tmp.indexes "$ENG_FILE" > cri_tmp.sample.eng
    awk 'NR==FNR{idx[$1]; next} FNR in idx' cri_tmp.indexes "$CRI_FILE" > cri_tmp.sample.cri

    cat ${additional_data_path}/cri/monolingual/watchtower_cri.syn_eng.txt \
        cri_tmp.sample.eng \
        > ${datadir}/$dataset/ours_all.eng-cri.eng_${LABEL}

    cat ${additional_data_path}/cri/monolingual/watchtower_cri.txt \
        cri_tmp.sample.cri \
        > ${datadir}/$dataset/ours_all.eng-cri.cri_${LABEL}

    rm cri_tmp.*

    for tgt in cri
    do
        src=eng
        echo $tgt
        if [[ $SPLIT_VAL_TEST == True ]]; then
        # split out 1,000 of new cri train data for validation 

            python ${git_repo_path}/scripts/data/split_val_test_data.py \
                --path1 $dataset/ours_all.${src}-${tgt}.${src}_${LABEL} \
                --path2 $dataset/ours_all.${src}-${tgt}.${tgt}_${LABEL}

            # copy our val data after split 
            cat ${datadir}/${dataset}/ours_all.${src}-${tgt}.${src}_${LABEL}_1000val \
                > ${datadir}/$dataset/validation_ours.${src}-${tgt}.${src}_${LABEL}

            cat ${datadir}/${dataset}/ours_all.${src}-${tgt}.${tgt}_${LABEL}_1000val \
                > ${datadir}/$dataset/validation_ours.${src}-${tgt}.${tgt}_${LABEL}

            # combine val sets 
            cat ${datadir}/${dataset}/validation_ours.${src}-${tgt}.${src}_${LABEL} \
                ${datadir}/${dataset}/validation_kmt.${src}-${tgt}.${src} \
                > ${datadir}/$dataset/validation.${src}-${tgt}.${src}_${LABEL}

            cat ${datadir}/${dataset}/validation_ours.${src}-${tgt}.${tgt}_${LABEL} \
                ${datadir}/${dataset}/validation_kmt.${src}-${tgt}.${tgt} \
                > ${datadir}/$dataset/validation.${src}-${tgt}.${tgt}_${LABEL}

        else
            # otherwise, just save kmt validation set as val set 
            cat ${datadir}/${dataset}/validation_kmt.${src}-${tgt}.${src} \
                > ${datadir}/$dataset/validation.${src}-${tgt}.${src}_${LABEL}

            cat ${datadir}/${dataset}/validation_kmt.${src}-${tgt}.${tgt} \
                > ${datadir}/$dataset/validation.${src}-${tgt}.${tgt}_${LABEL}
        fi

        # copy our training data (after split if splitting)
        cat ${datadir}/${dataset}/ours_all.${src}-${tgt}.${src}_${LABEL} \
            > ${datadir}/$dataset/train_ours.${src}-${tgt}.${src}_${LABEL}

        cat ${datadir}/${dataset}/ours_all.${src}-${tgt}.${tgt}_${LABEL} \
            > ${datadir}/$dataset/train_ours.${src}-${tgt}.${tgt}_${LABEL}

        # combine our train with kreyolmt train
        cat ${datadir}/${dataset}/train_ours.${src}-${tgt}.${src}_${LABEL} \
            ${datadir}/${dataset}/train_kmt.${src}-${tgt}.${src} \
            > ${datadir}/$dataset/train.${src}-${tgt}.${src}_${LABEL}

        cat ${datadir}/${dataset}/train_ours.${src}-${tgt}.${tgt}_${LABEL} \
            ${datadir}/${dataset}/train_kmt.${src}-${tgt}.${tgt} \
            > ${datadir}/$dataset/train.${src}-${tgt}.${tgt}_${LABEL}

    done 

    rm ${datadir}/${dataset}/ours_all.*

    # # clean training data and replace characters for eng-creoles
    for tgt in cri 
    do
        # filter the new training data sets 
        python ${git_repo_path}/scripts/data/clean_data.py \
            --path1 $dataset/train.${src}-${tgt}.${src}_${LABEL} \
            --path2 $dataset/train.${src}-${tgt}.${tgt}_${LABEL}
        files=("${datadir}/$dataset/train.eng-${tgt}.${tgt}_${LABEL}_filtered" \
                "${datadir}/$dataset/train.eng-${tgt}.eng_${LABEL}_filtered")
        for f in "${files[@]}"; 
        do
            echo $f
            sed -i "s/\“/@\"/g" $f
            sed -i "s/\”/調\"/g" $f
            sed -i "s/\“/付\"/g" $f
            sed -i "s/\’/혼\'/g" $f
            sed -i "s/\‘/ච\'/g" $f
            sed -i "s/\—/완\-/g" $f
            sed -i "s/\–/罪\-/g" $f
            sed -i "s/\«/\<\</g" $f
            sed -i "s/\»/\>\>/g" $f
            sed -i "s/\‚/\,/g" $f
        done

        python ${git_repo_path}/scripts/data/clean_data.py \
            --path1 $dataset/train_ours.${src}-${tgt}.${src}_${LABEL} \
            --path2 $dataset/train_ours.${src}-${tgt}.${tgt}_${LABEL}
        files=("${datadir}/$dataset/train_ours.eng-${tgt}.${tgt}_${LABEL}_filtered" \
                "${datadir}/$dataset/train_ours.eng-${tgt}.eng_${LABEL}_filtered")
        for f in "${files[@]}"; 
        do
            echo $f
            sed -i "s/\“/@\"/g" $f
            sed -i "s/\”/調\"/g" $f
            sed -i "s/\“/付\"/g" $f
            sed -i "s/\’/혼\'/g" $f
            sed -i "s/\‘/ච\'/g" $f
            sed -i "s/\—/완\-/g" $f
            sed -i "s/\–/罪\-/g" $f
            sed -i "s/\«/\<\</g" $f
            sed -i "s/\»/\>\>/g" $f
            sed -i "s/\‚/\,/g" $f
        done
        
        if [[ $SPLIT_VAL_TEST == True ]]; then
        # if we've also added val data, filter and clean new val sets:
            python ${git_repo_path}/scripts/data/clean_test_data.py \
                --path1 $dataset/validation.${src}-${tgt}.${src}_${LABEL} \
                --path2 $dataset/validation.${src}-${tgt}.${tgt}_${LABEL}
            files=("${datadir}/$dataset/validation.eng-${tgt}.${tgt}_${LABEL}_filtered" \
                    "${datadir}/$dataset/validation.eng-${tgt}.eng_${LABEL}_filtered")
            for f in "${files[@]}"; 
            do
                echo $f
                sed -i "s/\“/@\"/g" $f
                sed -i "s/\”/調\"/g" $f
                sed -i "s/\“/付\"/g" $f
                sed -i "s/\’/혼\'/g" $f
                sed -i "s/\‘/ච\'/g" $f
                sed -i "s/\—/완\-/g" $f
                sed -i "s/\–/罪\-/g" $f
                sed -i "s/\«/\<\</g" $f
                sed -i "s/\»/\>\>/g" $f
                sed -i "s/\‚/\,/g" $f
            done
            python ${git_repo_path}/scripts/data/clean_test_data.py \
                    --path1 $dataset/validation_ours.${src}-${tgt}.${src} \
                    --path2 $dataset/validation_ours.${src}-${tgt}.${tgt}
            files=("${datadir}/$dataset/validation_ours.eng-${tgt}.${tgt}_filtered" \
                    "${datadir}/$dataset/validation_ours.eng-${tgt}.eng_filtered")
            for f in "${files[@]}"; 
            do
                echo $f
                sed -i "s/\“/@\"/g" $f
                sed -i "s/\”/調\"/g" $f
                sed -i "s/\“/付\"/g" $f
                sed -i "s/\’/혼\'/g" $f
                sed -i "s/\‘/ච\'/g" $f
                sed -i "s/\—/완\-/g" $f
                sed -i "s/\–/罪\-/g" $f
                sed -i "s/\«/\<\</g" $f
                sed -i "s/\»/\>\>/g" $f
                sed -i "s/\‚/\,/g" $f
            done
        fi
    done

done 

    



