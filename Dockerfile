# Dockerfile

# ベースイメージの指定
# コマンドで使用されている NVIDIA TensorRT-LLM のリリースイメージを使用
FROM nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc5

# 必要なファイルをダウンロードするスクリプトを COPY コマンドの後に実行
# TIKTOKEN エンコーディングファイルを取得し、
# LLM API の追加設定ファイルを作成するスクリプト
COPY setup_and_run.sh /usr/local/bin/

# 実行スクリプトに実行権限を付与
RUN chmod +x /usr/local/bin/setup_and_run.sh
#RUN pip install --upgrade "nvidia-modelopt[hf]" --break-system-packages

RUN pip install torch openai
#RUN pip uninstall pynvml
RUN pip install nvidia-ml-py

# コンテナが起動したときに実行されるコマンド (CMD は docker-compose.yaml で上書きされる可能性あり)
# ENTRYPOINT を使用して、スクリプトを実行。これにより実行引数の処理が容易になる。
ENTRYPOINT ["/usr/local/bin/setup_and_run.sh"]

# TensorRT-LLM サービスがリッスンするポートを公開 (任意だが推奨)
EXPOSE 8355