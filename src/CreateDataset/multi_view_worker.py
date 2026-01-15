# Multi-View Worker script - processes all 6 camera views in a single API call
# Usage: python multi_view_worker.py <api_key_name> <artifact_folder> <output_path>
# 
# Artifact folder should contain 6 images with the naming pattern:
#   00_CAM_FRONT_*.jpg
#   01_CAM_FRONT_RIGHT_*.jpg
#   02_CAM_FRONT_LEFT_*.jpg
#   03_CAM_BACK_*.jpg
#   04_CAM_BACK_RIGHT_*.jpg
#   05_CAM_BACK_LEFT_*.jpg

import sys
import os
import re
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables from .env file
load_dotenv()

# Prefer a prompt file in the same directory; fall back across known filenames
PROMPT_CANDIDATES = (
    "system_prompt.txt",
    "system_prompt_summary.txt",
    "system_prompt_entitylist.txt",
)


def strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences (```json ... ```) from the response."""
    # Remove opening fence like ```json or ```
    text = re.sub(r'^```(?:json)?\s*\n?', '', text.strip())
    # Remove closing fence
    text = re.sub(r'\n?```\s*$', '', text)
    return text.strip()


def load_system_prompt(script_dir: Path) -> str:
    """Load the first available prompt file from known options."""
    for name in PROMPT_CANDIDATES:
        candidate = script_dir / name
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
    available = ", ".join([p.name for p in script_dir.glob("system_prompt*.txt")])
    raise FileNotFoundError(
        "No prompt file found. Expected one of: "
        f"{', '.join(PROMPT_CANDIDATES)}; available: {available or 'none'}"
    )

# View order and prefixes for 6 camera views
VIEW_PREFIXES = [
    "00_CAM_FRONT_",
    "01_CAM_FRONT_RIGHT_",
    "02_CAM_FRONT_LEFT_",
    "03_CAM_BACK_",
    "04_CAM_BACK_RIGHT_",
    "05_CAM_BACK_LEFT_",
]

VIEW_NAMES = [
    "Front",
    "Front Right",
    "Front Left",
    "Back",
    "Back Right",
    "Back Left",
]


def find_view_images(artifact_folder: Path) -> list[tuple[str, Path]]:
    """Find all 6 view images in the artifact folder and return them with their view names."""
    view_images = []
    
    for prefix, view_name in zip(VIEW_PREFIXES, VIEW_NAMES):
        # Find the file matching this prefix
        matching_files = list(artifact_folder.glob(f"{prefix}*.jpg"))
        if not matching_files:
            raise FileNotFoundError(f"No image found for view '{view_name}' with prefix '{prefix}' in {artifact_folder}")
        if len(matching_files) > 1:
            raise ValueError(f"Multiple images found for view '{view_name}' with prefix '{prefix}' in {artifact_folder}")
        
        view_images.append((view_name, matching_files[0]))
    
    return view_images


def generate(api_key_name: str, artifact_folder: str, output_path: str):
    # Get API key from environment (case-insensitive lookup)
    api_key = None
    for key in os.environ:
        if key.lower() == api_key_name.lower():
            api_key = os.environ.get(key)
            break
    
    if not api_key:
        print(f"ERROR: API key '{api_key_name}' not found in .env")
        sys.exit(1)
    
    artifact_path = Path(artifact_folder)
    if not artifact_path.exists():
        print(f"ERROR: Artifact folder '{artifact_folder}' not found")
        sys.exit(1)
    
    # Find all 6 view images
    try:
        view_images = find_view_images(artifact_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    client = genai.Client(api_key=api_key)

    # Load system prompt from the script directory with fallbacks
    system_prompt = load_system_prompt(Path(__file__).parent)

    # Build parts list with all 6 images and their labels
    parts = []
    
    for view_name, image_path in view_images:
        # Add the image with a label indicating which view it is
        image_data = image_path.read_bytes()
        parts.append(types.Part.from_text(text=f"[{view_name} View - Filename: {image_path.name}]"))
        parts.append(types.Part.from_bytes(data=image_data, mime_type="image/jpeg"))
    
    # Add the analysis instruction at the end
    parts.append(types.Part.from_text(text=f"{system_prompt}\n\nAnalyze this 360-degree driving scene from the 6 camera views provided above."))
    
    model = "gemini-robotics-er-1.5-preview"
    contents = [
        types.Content(
            role="user",
            parts=parts,
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=0,
        thinking_config=types.ThinkingConfig(
            thinking_budget=20000,
        ),
        media_resolution="MEDIA_RESOLUTION_HIGH",
    )

    # Collect response using streaming
    response_text = ""
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        response_text += chunk.text

    # Clean up markdown fences from response
    response_text = strip_markdown_fences(response_text)

    # Save output to file
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(response_text)
    
    print(f"SUCCESS: {artifact_path.name} -> {output_path}")
    print(f"  Processed views: {', '.join([v[0] for v in view_images])}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python multi_view_worker.py <api_key_name> <artifact_folder> <output_path>")
        print("")
        print("Arguments:")
        print("  api_key_name    - Name of the API key in .env file")
        print("  artifact_folder - Path to folder containing 6 camera view images")
        print("  output_path     - Path where the JSON response will be saved")
        sys.exit(1)
    
    api_key_name = sys.argv[1]
    artifact_folder = sys.argv[2]
    output_path = sys.argv[3]
    
    generate(api_key_name, artifact_folder, output_path)
