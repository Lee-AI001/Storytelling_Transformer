# Storytelling_Transformer

---

## LanternAI — A ongoing project aim at improving Transformer's storytelling ability :) 

Hey there!  
Welcome to **LanternAI**, a storytelling transformer designed to generate consistent, genre-rich stories ✍️🌌  
You're looking at version **v2.5**, and it's still growing, so feel free to explore, try it out, or even join the journey! 😄

---

## 🚀 What’s Inside?

- 💡 A transformer model trained for narrative generation  
- 📦 Pretrained model + tokenizer (already saved in `storage/`)  
- 🧠 Modular files: easy to understand and extend
-  :( unable to share the dataset cause size limit in github 

---

## 📁 Project Structure

/dataloader         # Handles input processing  
/model              # Transformer definition  
/storage/mid        # Where your models + logs live  
config.py           # All hyperparams and paths  
train.py            # Training pipeline  
pulling.py          # Load models and generate text  
z-plan.yaml         # info file 

after v3: 

daata_utils.py      # a separate file from orginal config to handle data process   
pretrain.py         # training for genres prediction 

---

## 👾 🚀 Versions

- Lantern_v2 - 
Basic Transformer setup,
Runs without errors,
Poor generation performance 

- Lantern_v2.5 -
Improved file structuring and modularization,
Added performance measurement tools and clearer logging,
No architectural/model upgrades

- Lantern_v3 - 
Implement regularization and normalization techniques such as embedding dropout, label smoothing, curriculum learning, and SGDR scheduling. Also, include an optional genre prediction feature (can be set to True or False).

- Lantern_v4 - 
Enhanced version with bug fixes. Add support for mixed precision training (bfloat16) and gradient accumulation steps to accelerate training.

- Lantern_v5 - 
Upgrade the user interface. Introduce an option for Multi-Head Latent Attention. Remove R-Drop and curriculum learning. Add pretraining epochs for genre classification and allow tuning of ROUGE test sample size.


---

## 🎉 Contribute / Say Hi

This project is just getting started and will keep evolving :)
If you want to add something, fix something, or just hang out — you’re super welcome here 🫶
Open a pull request or drop a star ⭐ if you like it!







