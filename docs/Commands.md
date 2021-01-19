PORPOSE Commands for TCGA Data
===========
# Training
### AMIL
``` shell
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_blca --mode path --model_type attention_mil
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_brca --mode path --model_type attention_mil
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_coadread --mode path --model_type attention_mil
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_gbmlgg --mode path --model_type attention_mil
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_hnsc --mode path --model_type attention_mil
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_kirc --mode path --model_type attention_mil
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_kirp --mode path --model_type attention_mil
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_lihc --mode path --model_type attention_mil
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_luad --mode path --model_type attention_mil
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_lusc --mode path --model_type attention_mil
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_paad --mode path --model_type attention_mil
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_skcm --mode path --model_type attention_mil
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_stad --mode path --model_type attention_mil
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_ucec --mode path --model_type attention_mil
```

### SNN
``` shell
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_blca --mode omic --reg_type omic --model_type max_net
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_brca --mode omic --reg_type omic --model_type max_net
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_coadread --mode omic --reg_type omic --model_type max_net
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_gbmlgg --mode omic --reg_type omic --model_type max_net
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_hnsc --mode omic --reg_type omic --model_type max_net
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_kirc --mode omic --reg_type omic --model_type max_net
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_kirp --mode omic --reg_type omic --model_type max_net
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_lihc --mode omic --reg_type omic --model_type max_net
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_luad --mode omic --reg_type omic --model_type max_net
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_lusc --mode omic --reg_type omic --model_type max_net
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_paad --mode omic --reg_type omic --model_type max_net
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_skcm --mode omic --reg_type omic --model_type max_net
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_stad --mode omic --reg_type omic --model_type max_net
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_ucec --mode omic --reg_type omic --model_type max_net
```

### MMF
``` shell
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_blca --mode pathomic --reg_type pathomic --model_type mm_attention_mil 
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_brca --mode pathomic --reg_type pathomic --model_type mm_attention_mil
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_coadread --mode pathomic --reg_type pathomic --model_type mm_attention_mil
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_gbmlgg --mode pathomic --reg_type pathomic --model_type mm_attention_mil
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_hnsc --mode pathomic --reg_type pathomic --model_type mm_attention_mil
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_kirc --mode pathomic --reg_type pathomic --model_type mm_attention_mil
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_kirp --mode pathomic --reg_type pathomic --model_type mm_attention_mil
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_lihc --mode pathomic --reg_type pathomic --model_type mm_attention_mil
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_luad --mode pathomic --reg_type pathomic --model_type mm_attention_mil
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_lusc --mode pathomic --reg_type pathomic --model_type mm_attention_mil
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_paad --mode pathomic --reg_type pathomic --model_type mm_attention_mil
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_skcm --mode pathomic --reg_type pathomic --model_type mm_attention_mil
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_stad --mode pathomic --reg_type pathomic --model_type mm_attention_mil
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_ucec --mode pathomic --reg_type pathomic --model_type mm_attention_mil
```

# Evaluation
### AMIL
``` shell
CUDA_VISIBLE_DEVICES=0 python eval_surv.py --which_splits 5foldcv --split_dir tcga_blca --mode path --model_type attention_mil
CUDA_VISIBLE_DEVICES=0 python eval_surv.py --which_splits 5foldcv --split_dir tcga_brca --mode path --model_type attention_mil
CUDA_VISIBLE_DEVICES=0 python eval_surv.py --which_splits 5foldcv --split_dir tcga_coadread --mode path --model_type attention_mil
CUDA_VISIBLE_DEVICES=0 python eval_surv.py --which_splits 5foldcv --split_dir tcga_gbmlgg --mode path --model_type attention_mil
CUDA_VISIBLE_DEVICES=0 python eval_surv.py --which_splits 5foldcv --split_dir tcga_hnsc --mode path --model_type attention_mil
CUDA_VISIBLE_DEVICES=0 python eval_surv.py --which_splits 5foldcv --split_dir tcga_kirc --mode path --model_type attention_mil
CUDA_VISIBLE_DEVICES=0 python eval_surv.py --which_splits 5foldcv --split_dir tcga_kirp --mode path --model_type attention_mil
CUDA_VISIBLE_DEVICES=0 python eval_surv.py --which_splits 5foldcv --split_dir tcga_lihc --mode path --model_type attention_mil
CUDA_VISIBLE_DEVICES=0 python eval_surv.py --which_splits 5foldcv --split_dir tcga_luad --mode path --model_type attention_mil
CUDA_VISIBLE_DEVICES=0 python eval_surv.py --which_splits 5foldcv --split_dir tcga_lusc --mode path --model_type attention_mil
CUDA_VISIBLE_DEVICES=0 python eval_surv.py --which_splits 5foldcv --split_dir tcga_paad --mode path --model_type attention_mil
CUDA_VISIBLE_DEVICES=0 python eval_surv.py --which_splits 5foldcv --split_dir tcga_skcm --mode path --model_type attention_mil
CUDA_VISIBLE_DEVICES=0 python eval_surv.py --which_splits 5foldcv --split_dir tcga_stad --mode path --model_type attention_mil
CUDA_VISIBLE_DEVICES=0 python eval_surv.py --which_splits 5foldcv --split_dir tcga_ucec --mode path --model_type attention_mil
```

### SNN
``` shell
CUDA_VISIBLE_DEVICES=0 python eval_surv.py --which_splits 5foldcv --split_dir tcga_blca --mode omic --reg_type omic --model_type max_net
CUDA_VISIBLE_DEVICES=0 python eval_surv.py --which_splits 5foldcv --split_dir tcga_brca --mode omic --reg_type omic --model_type max_net
CUDA_VISIBLE_DEVICES=0 python eval_surv.py --which_splits 5foldcv --split_dir tcga_coadread --mode omic --reg_type omic --model_type max_net
CUDA_VISIBLE_DEVICES=0 python eval_surv.py --which_splits 5foldcv --split_dir tcga_gbmlgg --mode omic --reg_type omic --model_type max_net
CUDA_VISIBLE_DEVICES=0 python eval_surv.py --which_splits 5foldcv --split_dir tcga_hnsc --mode omic --reg_type omic --model_type max_net
CUDA_VISIBLE_DEVICES=0 python eval_surv.py --which_splits 5foldcv --split_dir tcga_kirc --mode omic --reg_type omic --model_type max_net
CUDA_VISIBLE_DEVICES=0 python eval_surv.py --which_splits 5foldcv --split_dir tcga_kirp --mode omic --reg_type omic --model_type max_net
CUDA_VISIBLE_DEVICES=0 python eval_surv.py --which_splits 5foldcv --split_dir tcga_lihc --mode omic --reg_type omic --model_type max_net
CUDA_VISIBLE_DEVICES=0 python eval_surv.py --which_splits 5foldcv --split_dir tcga_luad --mode omic --reg_type omic --model_type max_net
CUDA_VISIBLE_DEVICES=0 python eval_surv.py --which_splits 5foldcv --split_dir tcga_lusc --mode omic --reg_type omic --model_type max_net
CUDA_VISIBLE_DEVICES=0 python eval_surv.py --which_splits 5foldcv --split_dir tcga_paad --mode omic --reg_type omic --model_type max_net
CUDA_VISIBLE_DEVICES=0 python eval_surv.py --which_splits 5foldcv --split_dir tcga_skcm --mode omic --reg_type omic --model_type max_net
CUDA_VISIBLE_DEVICES=0 python eval_surv.py --which_splits 5foldcv --split_dir tcga_stad --mode omic --reg_type omic --model_type max_net
CUDA_VISIBLE_DEVICES=0 python eval_surv.py --which_splits 5foldcv --split_dir tcga_ucec --mode omic --reg_type omic --model_type max_net
```

### MMF
``` shell
CUDA_VISIBLE_DEVICES=0 python eval_surv.py --which_splits 5foldcv --split_dir tcga_blca --mode pathomic --reg_type pathomic --model_type mm_attention_mil 
CUDA_VISIBLE_DEVICES=0 python eval_surv.py --which_splits 5foldcv --split_dir tcga_brca --mode pathomic --reg_type pathomic --model_type mm_attention_mil
CUDA_VISIBLE_DEVICES=0 python eval_surv.py --which_splits 5foldcv --split_dir tcga_coadread --mode pathomic --reg_type pathomic --model_type mm_attention_mil
CUDA_VISIBLE_DEVICES=0 python eval_surv.py --which_splits 5foldcv --split_dir tcga_gbmlgg --mode pathomic --reg_type pathomic --model_type mm_attention_mil
CUDA_VISIBLE_DEVICES=0 python eval_surv.py --which_splits 5foldcv --split_dir tcga_hnsc --mode pathomic --reg_type pathomic --model_type mm_attention_mil
CUDA_VISIBLE_DEVICES=0 python eval_surv.py --which_splits 5foldcv --split_dir tcga_kirc --mode pathomic --reg_type pathomic --model_type mm_attention_mil
CUDA_VISIBLE_DEVICES=0 python eval_surv.py --which_splits 5foldcv --split_dir tcga_kirp --mode pathomic --reg_type pathomic --model_type mm_attention_mil
CUDA_VISIBLE_DEVICES=0 python eval_surv.py --which_splits 5foldcv --split_dir tcga_lihc --mode pathomic --reg_type pathomic --model_type mm_attention_mil
CUDA_VISIBLE_DEVICES=0 python eval_surv.py --which_splits 5foldcv --split_dir tcga_luad --mode pathomic --reg_type pathomic --model_type mm_attention_mil
CUDA_VISIBLE_DEVICES=0 python eval_surv.py --which_splits 5foldcv --split_dir tcga_lusc --mode pathomic --reg_type pathomic --model_type mm_attention_mil
CUDA_VISIBLE_DEVICES=0 python eval_surv.py --which_splits 5foldcv --split_dir tcga_paad --mode pathomic --reg_type pathomic --model_type mm_attention_mil
CUDA_VISIBLE_DEVICES=0 python eval_surv.py --which_splits 5foldcv --split_dir tcga_skcm --mode pathomic --reg_type pathomic --model_type mm_attention_mil
CUDA_VISIBLE_DEVICES=0 python eval_surv.py --which_splits 5foldcv --split_dir tcga_stad --mode pathomic --reg_type pathomic --model_type mm_attention_mil
CUDA_VISIBLE_DEVICES=0 python eval_surv.py --which_splits 5foldcv --split_dir tcga_ucec --mode pathomic --reg_type pathomic --model_type mm_attention_mil
```
