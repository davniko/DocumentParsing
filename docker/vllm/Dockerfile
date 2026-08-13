# syntax=docker/dockerfile:1.7@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e

FROM vllm/vllm-openai:v0.26.0@sha256:ffb2d59b1c059a5bd8d781320c9f5189de8293693b7d95da54befddaa54abf52

LABEL org.opencontainers.image.source="https://github.com/vllm-project/vllm" \
      org.opencontainers.image.revision="df63cb9492e85d3df71284a5d9f234fc39ae0b74" \
      org.opencontainers.image.base.name="vllm/vllm-openai:v0.26.0" \
      org.opencontainers.image.base.digest="sha256:ffb2d59b1c059a5bd8d781320c9f5189de8293693b7d95da54befddaa54abf52" \
      org.document-ocr.vllm.patch.49869.commit="89e3c3f5b41d0f678d19a138dba59b3757a1a16f" \
      org.document-ocr.vllm.patch.49869.sha256="0923d3e1975634a7e2b639119a8d78a190a991c5635fba7b8e2df7d17b5e3993" \
      org.document-ocr.vllm.patch.51966.commit="df63cb9492e85d3df71284a5d9f234fc39ae0b74" \
      org.document-ocr.vllm.patch.51966.sha256="fbc1648c49f56ae4d3ee664969672afdfe1230da5f4878f285bbd99ad6469588" \
      org.document-ocr.vllm.build-manifest.sha256="c1ffd97e7a6fa3fdb02b5acd84966e6afc10c096a5eec5e7bd47dad7a88197b4"

COPY --chmod=0444 docker/vllm/build-manifest.json /opt/document-ocr/vllm-build-manifest.json
COPY --chmod=0444 docker/vllm/patches/49869-glm-ocr-mtp-weight-prefix.patch /tmp/vllm-mtp-weight-prefix.patch
COPY --chmod=0444 docker/vllm/patches/51966-glm-ocr-mtp-cudagraph.patch /tmp/vllm-mtp-cudagraph.patch

RUN set -eux; \
    echo "c1ffd97e7a6fa3fdb02b5acd84966e6afc10c096a5eec5e7bd47dad7a88197b4  /opt/document-ocr/vllm-build-manifest.json" | sha256sum -c -; \
    echo "0923d3e1975634a7e2b639119a8d78a190a991c5635fba7b8e2df7d17b5e3993  /tmp/vllm-mtp-weight-prefix.patch" | sha256sum -c -; \
    echo "fbc1648c49f56ae4d3ee664969672afdfe1230da5f4878f285bbd99ad6469588  /tmp/vllm-mtp-cudagraph.patch" | sha256sum -c -; \
    echo "f06d1a1a8d92e6ab39ebf7931d3eb3d6688d212606b1c409d7bed8eed5ba1fc2  /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/utils.py" | sha256sum -c -; \
    echo "f7e5c90fdefdad05f05cad17cc74c7f360a349aeecfb3885212b4f07dd8d1fc1  /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/glm_ocr_mtp.py" | sha256sum -c -; \
    echo "a4c0fab8092b04fc0c54b4c8936ba4712a69d734f82bd6018572238e828cc444  /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/glm4_moe_lite_mtp.py" | sha256sum -c -; \
    patch --batch --fuzz=0 --forward -p1 -d /usr/local/lib/python3.12/dist-packages < /tmp/vllm-mtp-weight-prefix.patch; \
    patch --batch --fuzz=0 --forward -p1 -d /usr/local/lib/python3.12/dist-packages < /tmp/vllm-mtp-cudagraph.patch; \
    echo "f8d69922ac178e4e8dc8083bef8714234005254785c2a3645ed74b3dfe521733  /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/utils.py" | sha256sum -c -; \
    echo "034d815af1c797bd2d7437d799efe0bc8010ab1cf8cfe8a3f39b3c89158f6101  /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/glm_ocr_mtp.py" | sha256sum -c -; \
    echo "499d3f66c7a939991bad6e760ba5a396fd49f32bab054fec1b440a4cc8408529  /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/glm4_moe_lite_mtp.py" | sha256sum -c -; \
    rm /tmp/vllm-mtp-weight-prefix.patch /tmp/vllm-mtp-cudagraph.patch

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

safe_masks = {
    Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/glm_ocr_mtp.py"): (
        "positions[0].unsqueeze(-1) == 0, 0, inputs_embeds",
        "inputs_embeds[positions[0] == 0] = 0",
    ),
    Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/glm4_moe_lite_mtp.py"): (
        "torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)",
        "inputs_embeds[positions == 0] = 0",
    ),
}
for path, (required, forbidden) in safe_masks.items():
    source = path.read_text()
    ast.parse(source, filename=str(path))
    assert required in source
    assert forbidden not in source
PY
