# Source Attribution:
# The majority of the code is derived from the following source:
# - LLaMA-Factory GitHub Repository: https://github.com/hiyouga/LLaMA-Factory
# - Tag: v0.9.0
# - File: /src/webui.py
# - Reference Paper: LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models (Zheng et al., ACL 2024)

import os

from llama_factory.webui.interface import create_ui


def main():
    gradio_share = os.environ.get("GRADIO_SHARE", "0").lower() in ["true", "1"]
    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = os.environ.get("GRADIO_SERVER_PORT", 9876)
    create_ui().queue().launch(share=gradio_share, server_name=server_name, server_port=server_port, inbrowser=True)


if __name__ == "__main__":
    main()
