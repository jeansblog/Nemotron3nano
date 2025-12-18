docker run --rm -it --gpus=all nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc5 nvidia-smi

docker run --rm -it --gpus all \
  nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc5 \
  python -c "import tensorrt_llm; print(f'TensorRT-LLM version: {tensorrt_llm.__version__}')"

[TensorRT-LLM] TensorRT LLM version: 1.2.0rc5
TensorRT-LLM version: 1.2.0rc5

export MODEL_HANDLE="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"

docker run --name trtllm_llm_server --rm -it --gpus all --ipc host --network host   -e HF_TOKEN=$HF_TOKEN   -e MODEL_HANDLE="$MODEL_HANDLE"   -v $HOME/.cache/huggingface/:/root/.cache/huggingface/   nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc5  bash -c '
    hf download $MODEL_HANDLE && \
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
 '