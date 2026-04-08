# Full pipeline
python -m MCDataGen.run_pipeline --num-rows 10000


MCDataGen/
__init__.py
config.py
prompts_mc.py
utils.py ← SONL I/O + resume helpers
phase1_mc_gen.py ← generation + inline evaluation
phase2_mc_eval.py ← LLM judge
run_pipeline.py