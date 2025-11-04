## CMU 18734/17731 Project Phase 2 and 3 Repo

## Overview
We provide three model sets — each with a target model and its own test set. The MIA (membership inference attack) difficulty levels of the three sets are similar, but you may see small differences in MIA performance across them even when using the same attack code.

We release full details of the train set, including the training data and test labels. This lets you evaluate your method locally and get an estimate of how it will perform when you submit scores for the val and final sets. You can view real-time scores for the validation set, though submissions are time-limited. Final-set scores are only revealed when the final set is opened at the end of the semester and will determine the final rankings. Please develop a general attack method rather than overfitting to the validation set to chase leaderboard performance — the final set decides ranking, and the validation set serves as an additional verification.

## Evaluation
We report **TPR @ 0.01 FPR**.  
- **Phase 2** has a minimum requirement of **0.15** — you must exceed 0.15 to receive that portion of the score.  
- **Phase 3** raises the requirement (not decided yet); we recommend aiming as high as possible (e.g., TPR > 0.35).
- If your MIA techniques for phase 2 already outperform the TPR requirement for phase 3, you may reuse your code for phase 3 submission. 

**Hint:** You have access to the in-distribution data and may use it to train shadow models. Proper use of these data for shadow modeling can greatly improve your MIA score!!!

## Folder structure
- `data/` — contains three subfolders: `train`, `val`, `final`.
  - `train/` includes training data, test data, and test labels.
  - `val/` and `final/` include only test data; your task is to predict membership for these sets.
  - `prepare_data.py` — script used to collect and inspect the training data. Use it to understand the data and to prepare data for shadow models.
- `ft_llm/`
  - `ft_llm.py` — script for fine-tuning a shadow model.
  - `ft_llm_colab.py` — script for fine-tuning a shadow model on colab T4 GPU (T4 does not natively support bf16).
  - `ft_llm.sh` — bash script with our finetuning configuration.
- `models/` — contains three model directories: `train`, `val`, `final`. Use the `train` model for experiments, then predict membership for `val` and `final`.
- `MIA_phase2_3.ipynb` — starter kit for phase 2 and phase 3. Both phases use the same model but have different performance requirements.

## Result submission
participate by your andrew email and submit the zip file (as specificied by notebook) to val phase. The leaderboard is based on codabench, link: https://www.codabench.org/competitions/11238/

Please submit by groups after forming an organization (clike your name on the top right and create an organization).

Please use andrew email to register only. If you have any questions, email me or post a discussion on canvas.

## Clarification (Newly added 11.3)

As some of you have asked about the details of the project, we confirm some settings here:
 - All three models are trained with LORA, the same setting `-m gpt2 --block_size 512 --epochs 3 --batch_size 8 --gradient_accumulation_steps 1 --lr 2e-4 --lora --lora_r 32 --lora_alpha 64 --lora_dropout 0.05`. The only differences are the training data. (10k training data for each but 3 different sets).
 - You may find out that train phase training data is actually achieved by setting `seed=42` in "data/prepare_data.py". To train other shadow models, remember to change the seed number.
 - I suggest you train your shadow models with the same config to start, once you achieved like 10% TPR at 1% FPR, you may do the hyperparameter tuning and other training configs that you feel reasonable.


## Report
stay tuned for the report content and format.


