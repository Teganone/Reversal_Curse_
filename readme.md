# Readme

## env
env: `pip install -r requirements.txt `

the python version should be >= 3.10

## code

`src/bart.py`: fine-tune bart
`src/bart_eval.py`: evaluate bart model for experiment 1
`src/bert_exp2.py`: evaluate reverse relationship for experiment2
`src/llm_exp2.py`: modify query of gpt-4 for experiment2
`src/draw_plot_bar.ipynb`: draw the results of experiment1

I use the azureopenai, which require the `openai_api_key`. If you'd like to run the code for experiment2 properly, you have to change some code concerning openai. Thanks.

`data/celebrity_relations/parent_child_pairs_bert.csv`: restores the results of `bert_exp2.py`
`data/celebrity_relations/parent_child_pairs_llm_which_prompt.csv`: restores the results of `llm_exp2.py`
`data/reverse_experiments/june_version_7921032488/results/bart_sim_acc_summary_base.csv`: restores the results of experiment1