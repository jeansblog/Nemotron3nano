#!/bin/bash
# setup_and_run.sh

set -e # エラーが発生した場合は直ちに終了

# MODEL_HANDLEが設定されていることを確認
if [ -z "$MODEL_HANDLE" ]; then
    echo "Error: MODEL_HANDLE environment variable is not set." >&2
    exit 1
fi

# Hugging Face モデルをダウンロード
echo "Downloading Hugging Face model: $MODEL_HANDLE..."
hf download "$MODEL_HANDLE"

# LLM API の追加設定ファイルを作成
echo "Creating nano_v3.yaml..."
cat > nano_v3.yaml<<EOF
runtime: trtllm
compile_backend: torch-cudagraph
max_batch_size: 64
max_seq_len: 16384
enable_chunked_prefill: true
attn_backend: flashinfer
model_factory: AutoModelForCausalLM
skip_loading_weights: false
free_mem_ratio: 0.65
cuda_graph_batch_sizes: [1, 2, 4, 8, 16, 24, 32, 64, 128, 256, 320, 384]
kv_cache_config:
  # disable kv_cache reuse since not supported for hybrid/ssm models
  enable_block_reuse: false
transforms:
  detect_sharding:
    sharding_dims: ['ep', 'bmm']
    allreduce_strategy: 'AUTO'
    manual_config:
      head_dim: 128
      tp_plan:
        # mamba SSM layer
        "in_proj": "mamba"
        "out_proj": "rowwise"
        # attention layer
        "q_proj": "colwise"
        "k_proj": "colwise"
        "v_proj": "colwise"
        "o_proj": "rowwise"
        # NOTE: consider not sharding shared experts and/or
        # latent projections at all, keeping them replicated.
        # To do so, comment out the corresponding entries.
        # moe layer: SHARED experts
        "up_proj": "colwise"
        "down_proj": "rowwise"
        # MoLE: latent projections: simple shard
        "fc1_latent_proj": "gather"
        "fc2_latent_proj": "gather"
  multi_stream_moe:
    stage: compile
    enabled: true
  insert_cached_ssm_attention:
      cache_config:
        mamba_dtype: float32
  fuse_mamba_a_log:
    stage: post_load_fusion
    enabled: true
EOF
echo "Starting trtllm-serve for model $MODEL_HANDLE on port 8355..."
TRTLLM_ENABLE_PDL=1 trtllm-serve "$MODEL_HANDLE" \
  --backend _autodeploy \
  --trust_remote_code \
  --reasoning_parser deepseek-r1 \
  --extra_llm_api_options nano_v3.yaml \
  --port 8355 \
  --host 0.0.0.0 \