import base64
import hashlib
from io import BytesIO

import torch
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np

# Import your custom pipeline class
from pipeline_flux_rf_inversion_update import RFInversionFluxPipeline
from diffusers.utils import load_image

app = Flask(__name__)

# Define some aspect ratios (same as in your original code)
ASPECT_RATIOS = {
    "1:1": (1024, 1024),
    "16:9": (1344, 768),
    "21:9": (1536, 640),
    "3:2": (1216, 832),
    "2:3": (832, 1216),
    "4:5": (896, 1088),
    "5:4": (1088, 896),
    "3:4": (896, 1152),
    "4:3": (1152, 896),
    "9:16": (768, 1344),
    "9:21": (640, 1536),
}

def get_closest_aspect_ratio(width, height, aspect_ratios):
    """Find the aspect ratio from ASPECT_RATIOS closest to the input image's aspect ratio."""
    aspect_ratio = width / height
    closest_ratio = None
    closest_size = None
    min_diff = float("inf")
    for ratio, (w, h) in aspect_ratios.items():
        ratio_value = w / h
        diff = abs(ratio_value - aspect_ratio)
        if diff < min_diff:
            min_diff = diff
            closest_ratio = ratio
            closest_size = (w, h)
    return closest_ratio, closest_size

def decode_image(base64_string):
    """Decode a base64-encoded string into a PIL image (RGB)."""
    image_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(image_data)).convert("RGB")

# ------------------------------------------------
# Load the Flex model pipeline on container start
# ------------------------------------------------
print("Loading RFInversionFluxPipeline model...")
pipe = RFInversionFluxPipeline.from_pretrained(
    "ostris/Flex.1-alpha",
    torch_dtype=torch.bfloat16
)
pipe.to("cuda")
print("Model loaded successfully.")

@app.route("/infer", methods=["POST"])
def infer():
    """
    Expects JSON:
    {
      "image": "<base64-encoded image>",
      "mask": "<base64-encoded mask>",
      "num_inference_steps_inversions": 50,
      "gamma_white": 0.8,
      "gamma_black": 0.9,
      "prompt_text": "",
      "prompt_2": "",
      "start_timestep": 0.02,
      "stop_timestep": 0.2,
      "num_inference_steps": 50,
      "eta_white": 0.7,
      "eta_black": 0.9,
      "guidance_scale": 5
    }
    """
    data = request.get_json()

    # Required: image + mask in base64
    image_b64 = data.get("image")
    mask_b64 = data.get("mask")
    if not image_b64 or not mask_b64:
        return jsonify({"error": "Missing 'image' or 'mask' in request"}), 400

    try:
        # Decode images
        image = decode_image(image_b64)
        mask = decode_image(mask_b64)

        # Determine closest aspect ratio to the input image
        width, height = image.size
        _, (new_width, new_height) = get_closest_aspect_ratio(width, height, ASPECT_RATIOS)

        # Gather inversion parameters (or use defaults)
        num_inference_steps_inversions = data.get("num_inference_steps_inversions", 50)
        gamma_white = data.get("gamma_white", 0.8)
        gamma_black = data.get("gamma_black", 0.9)

        # Gather generation/editing parameters (or use defaults)
        prompt_text = data.get("prompt_text", "")
        prompt_2 = data.get("prompt_2", "")
        start_timestep = data.get("start_timestep", 1/50)
        stop_timestep = data.get("stop_timestep", 10/50)
        num_inference_steps = data.get("num_inference_steps", 50)
        eta_white = data.get("eta_white", 0.7)
        eta_black = data.get("eta_black", 0.9)
        guidance_scale = data.get("guidance_scale", 5)

        # -------------------
        # Perform inversion
        # -------------------
        inverted_latents, image_latents, latent_image_ids, mask_latents = pipe.invert(
            image=image,
            mask_image=mask,
            source_prompt="",
            num_inversion_steps=num_inference_steps_inversions,
            gamma_white=gamma_white,
            gamma_black=gamma_black,
            height=new_height,
            width=new_width,
        )

        # --------------------
        # Perform generation
        # --------------------
        with torch.no_grad():
            output = pipe(
                prompt=prompt_text,
                prompt_2=prompt_2,
                inverted_latents=inverted_latents,
                image_latents=image_latents,
                latent_image_ids=latent_image_ids,
                mask_latents=mask_latents,
                start_timestep=start_timestep,
                stop_timestep=stop_timestep,
                num_inference_steps=num_inference_steps,
                height=new_height,
                width=new_width,
                eta_white=eta_white,
                eta_black=eta_black,
                guidance_scale=guidance_scale,
            )
            edited_image = output.images[0]

        # Convert edited image to base64
        buffered = BytesIO()
        edited_image.save(buffered, format="PNG")
        edited_image_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Optional: generate a unique filename
        unique_hash = hashlib.md5(prompt_text.encode()).hexdigest()[:8]
        filename = f"edited_{unique_hash}.png"

        return jsonify({
            "filename": filename,
            "image": edited_image_b64
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
