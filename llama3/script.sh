# show ollama model info
# ollama show llama3.2:1b --modelfile

# rag system fine tune Guide by ollama
# ollama show llama3.2:1b --modelfile > custom.modelfile
# vi custom.modelfile
# ollama create gman_rag --file ./custom.modelfile
# ollama run 

# python3 lora.py --train --model meta-llama/Llama-3.2-1B --data ./data --batch-size 1 --lora-layers 4 --iters 1000

# meta-llama/Llama-3.2-1B

# https://medium.com/@kramiknakrani100/fine-tune-qwen-2-5-on-custom-data-using-free-google-colab-a-step-by-step-guide-99281ef228f8