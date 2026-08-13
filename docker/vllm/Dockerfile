# syntax=docker/dockerfile:1.7@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e

FROM vllm/vllm-openai:v0.26.0@sha256:ffb2d59b1c059a5bd8d781320c9f5189de8293693b7d95da54befddaa54abf52

LABEL org.opencontainers.image.source="https://github.com/vllm-project/vllm/pull/49869" \
      org.opencontainers.image.revision="89e3c3f5b41d0f678d19a138dba59b3757a1a16f" \
      org.opencontainers.image.base.name="vllm/vllm-openai:v0.26.0" \
      org.opencontainers.image.base.digest="sha256:ffb2d59b1c059a5bd8d781320c9f5189de8293693b7d95da54befddaa54abf52" \
      org.document-ocr.vllm.patch.sha256="0923d3e1975634a7e2b639119a8d78a190a991c5635fba7b8e2df7d17b5e3993" \
      org.document-ocr.vllm.build-manifest.sha256="02f79af13ea66b4c63cf1609abc9287a7757efe9b7dd03ddeeb0d73ed041352d"

COPY --chmod=0444 docker/vllm/build-manifest.json /opt/document-ocr/vllm-build-manifest.json
COPY --chmod=0444 docker/vllm/patches/49869-glm-ocr-mtp-weight-prefix.patch /tmp/vllm-mtp.patch

RUN set -eux; \
    echo "02f79af13ea66b4c63cf1609abc9287a7757efe9b7dd03ddeeb0d73ed041352d  /opt/document-ocr/vllm-build-manifest.json" | sha256sum -c -; \
    echo "0923d3e1975634a7e2b639119a8d78a190a991c5635fba7b8e2df7d17b5e3993  /tmp/vllm-mtp.patch" | sha256sum -c -; \
    echo "f06d1a1a8d92e6ab39ebf7931d3eb3d6688d212606b1c409d7bed8eed5ba1fc2  /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/utils.py" | sha256sum -c -; \
    patch --batch --fuzz=0 --forward -p1 -d /usr/local/lib/python3.12/dist-packages < /tmp/vllm-mtp.patch; \
    echo "f8d69922ac178e4e8dc8083bef8714234005254785c2a3645ed74b3dfe521733  /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/utils.py" | sha256sum -c -; \
    rm /tmp/vllm-mtp.patch

RUN PYTHONDONTWRITEBYTECODE=1 python3 - <<'PY'
import ast
from pathlib import Path
from types import SimpleNamespace

target = Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/utils.py")
tree = ast.parse(target.read_bytes(), filename=str(target))
function = next(
    node
    for node in tree.body
    if isinstance(node, ast.FunctionDef) and node.name == "get_spec_layer_idx_from_weight_name"
)
namespace = {}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(target), "exec"), namespace)
classify = namespace["get_spec_layer_idx_from_weight_name"]

mtp = SimpleNamespace(num_hidden_layers=16, num_nextn_predict_layers=1)
without_mtp = SimpleNamespace(num_hidden_layers=16)

assert classify(mtp, "model.layers.16.x") == 16
assert classify(mtp, "layers.16.x") == 16
assert classify(mtp, "model.language_model.layers.16.x") == 16
assert classify(mtp, "model.language_model.layers.15.x") is None
assert classify(without_mtp, "model.language_model.layers.16.x") is None
PY
