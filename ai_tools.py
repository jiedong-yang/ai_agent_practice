import replicate
import os
import requests
import json
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from typing import List, Any, Dict, Optional, Literal
from pydantic_ai import Agent
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

load_dotenv()

# Create output directory
OUTPUT_DIR = Path("generated_images")
OUTPUT_DIR.mkdir(exist_ok=True)

# Define allowed aspect ratios (from Flux Kontext model)
AllowedAspectRatio = Literal[
    "match_input_image", "1:1", "16:9", "9:16", "4:3", "3:4",
    "3:2", "2:3", "5:4", "4:5", "21:9", "9:21", "2:1", "1:2"
]


class ImageGenParams(BaseModel):
    prompt: str = Field(description="Image prompt for image generation model")
    input_image: Optional[str] = Field(default=None, description="Path or url to input image for img2img or None")
    aspect_ratio: AllowedAspectRatio = Field(default="1:1", description="Image aspect ratio from allowed values")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    output_format: str = Field(default="png", description="Output image format")
    safety_tolerance: int = Field(default=2, ge=0, le=6, description="Safety filter level 0-6")

    @field_validator('aspect_ratio', mode='before')
    @classmethod
    def validate_aspect_ratio(cls, v):
        """Convert common aspect ratio descriptions to allowed values"""
        if v is None:
            return "1:1"

        aspect_str = str(v).lower().strip()

        # Direct matches first
        allowed_ratios = [
            "match_input_image", "1:1", "16:9", "9:16", "4:3", "3:4",
            "3:2", "2:3", "5:4", "4:5", "21:9", "9:21", "2:1", "1:2"
        ]
        if v in allowed_ratios:
            return v

        # Comprehensive mapping
        aspect_mappings = {
            # Special case
            "match_input_image": "match_input_image",
            "match": "match_input_image",
            "same as input": "match_input_image",

            # Square formats (1:1)
            "square": "1:1", "1:1": "1:1", "1x1": "1:1", "equal": "1:1",
            "instagram": "1:1", "profile": "1:1", "avatar": "1:1", "logo": "1:1", "icon": "1:1",

            # Landscape formats (wider than tall)
            "landscape": "16:9", "16:9": "16:9", "16x9": "16:9", "widescreen": "16:9",
            "wide": "16:9", "horizontal": "16:9", "cinematic": "16:9", "panorama": "16:9",

            # Portrait formats (taller than wide)
            "portrait": "3:4", "3:4": "3:4", "3x4": "3:4", "vertical": "3:4",
            "tall": "3:4", "upright": "3:4", "book": "3:4", "poster": "3:4",

            # Mobile portrait (very tall)
            "mobile": "9:16", "phone": "9:16", "9:16": "9:16", "9x16": "9:16",
            "story": "9:16", "tiktok": "9:16", "reel": "9:16",

            # Traditional photo/TV ratios
            "4:3": "4:3", "4x3": "4:3", "traditional": "4:3", "tv": "4:3", "monitor": "4:3",

            # Film/photo ratios
            "3:2": "3:2", "3x2": "3:2", "35mm": "3:2", "film": "3:2", "photo": "3:2",
            "2:3": "2:3", "2x3": "2:3",

            # Slightly wider/taller squares
            "5:4": "5:4", "5x4": "5:4", "4:5": "4:5", "4x5": "4:5",

            # Ultra-wide formats
            "21:9": "21:9", "21x9": "21:9", "ultrawide": "21:9", "cinema": "21:9", "anamorphic": "21:9",
            "9:21": "9:21", "9x21": "9:21",

            # Double ratios
            "2:1": "2:1", "2x1": "2:1", "panoramic": "2:1",
            "1:2": "1:2", "1x2": "1:2",
        }

        # Check mappings
        if aspect_str in aspect_mappings:
            return aspect_mappings[aspect_str]

        # Check for partial matches
        for term, ratio in aspect_mappings.items():
            if term in aspect_str:
                return ratio

        return "1:1"  # Default fallback


def extract_replicate_id_from_url(url: str) -> str:
    """Extract prediction ID from Replicate URL for consistent naming"""
    try:
        # Replicate URLs typically look like:
        # https://replicate.delivery/pbxt/abc123def456.png
        # https://replicate.delivery/mgxm/xyz789.webp
        filename = url.split('/')[-1]
        prediction_id = filename.split('.')[0]

        # Ensure it's a valid-looking ID (alphanumeric, reasonable length)
        if prediction_id and len(prediction_id) >= 8 and prediction_id.isalnum():
            return prediction_id
        else:
            raise ValueError("Invalid prediction ID format")

    except:
        # Fallback to timestamp if extraction fails
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:17]


def save_image_with_metadata(
        image_data: bytes,
        original_url: str,
        prompt: str,
        params: ImageGenParams,
        custom_name: str = None
) -> Dict[str, str]:
    """
    Save image with consistent naming and comprehensive metadata
    """
    try:
        # Extract prediction ID from Replicate URL
        if custom_name:
            base_name = custom_name
        else:
            base_name = extract_replicate_id_from_url(original_url)

        # Create filename
        filename = f"{base_name}.{params.output_format}"
        filepath = OUTPUT_DIR / filename

        # Save image
        with open(filepath, 'wb') as f:
            f.write(image_data)

        # Create comprehensive metadata
        metadata = {
            "filename": filename,
            "filepath": str(filepath),
            "original_url": original_url,
            "prompt": prompt,
            "parameters": params.model_dump(),
            "generated_at": datetime.now().isoformat(),
            "prediction_id": base_name,
            "file_size_bytes": len(image_data),
            "output_directory": str(OUTPUT_DIR)
        }

        # Save metadata as JSON
        metadata_filename = f"{base_name}_metadata.json"
        metadata_filepath = OUTPUT_DIR / metadata_filename
        with open(metadata_filepath, 'w') as f:
            json.dump(metadata, f, indent=2)

        return {
            "filepath": str(filepath),
            "prediction_id": base_name,
            "original_url": original_url,
            "metadata_file": str(metadata_filepath)
        }

    except Exception as e:
        raise Exception(f"Error saving image with metadata: {str(e)}")


async def formatting_params_from_prompt(prompt: str, input_image_url: str = None) -> ImageGenParams:
    """Enhanced parameter extraction with all 14 allowed aspect ratios and img2img support"""

    # If input_image_url is provided, extract it from the prompt or use directly
    detected_input_image = input_image_url

    if not detected_input_image:
        # Look for URLs in the prompt
        import re
        url_pattern = r'https?://[^\s]+'
        urls = re.findall(url_pattern, prompt)
        if urls:
            detected_input_image = urls[0]
            # Clean up the prompt by removing the URL
            prompt = re.sub(url_pattern, '', prompt).strip()

    agent = Agent(
        'google-gla:gemini-2.5-flash',
        system_prompt=f"""
        Extract image generation parameters from the input prompt and return them in the specified format.

        INPUT IMAGE HANDLING:
        - If an input image is provided, this enables img2img transformation capabilities
        - Input image URL/path: {detected_input_image if detected_input_image else 'None'}
        - For img2img with "match aspect ratio" requests, use "match_input_image" as aspect_ratio

        CRITICAL ASPECT RATIO RULES:
        - ONLY use these exact values: "match_input_image", "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "5:4", "4:5", "21:9", "9:21", "2:1", "1:2"
        - If prompt mentions "match aspect ratio" or "same aspect ratio" AND input image exists → use "match_input_image"

        ASPECT RATIO SELECTION GUIDE:
        - "match_input_image" = Match the input image dimensions (for img2img)
        - "1:1" = Square (social media posts, logos, icons, profile pictures)
        - "16:9" = Landscape widescreen (scenery, cinematic shots, desktop wallpapers)
        - "9:16" = Portrait mobile (phone wallpapers, TikTok, Instagram stories)
        - "4:3" = Traditional landscape (TV, monitors, classic photos)
        - "3:4" = Traditional portrait (book covers, posters, standing people)
        - "3:2" = Film/photo landscape (35mm film ratio, DSLR photos)
        - "2:3" = Film/photo portrait (35mm film portrait)
        - "5:4" = Slightly wide square
        - "4:5" = Slightly tall square
        - "21:9" = Ultra-wide cinematic (anamorphic, ultra-wide monitors)
        - "9:21" = Ultra-tall portrait
        - "2:1" = Panoramic landscape
        - "1:2" = Tall panoramic portrait

        ASPECT RATIO KEYWORD DETECTION:
        Square (1:1): "square", "logo", "icon", "instagram post", "profile picture"
        Landscape (16:9): "landscape", "widescreen", "cinematic", "horizontal", "wide"
        Portrait mobile (9:16): "mobile", "phone", "story", "tiktok", "reel", "vertical phone"
        Traditional landscape (4:3): "traditional", "tv", "monitor", "classic"
        Traditional portrait (3:4): "portrait", "book cover", "poster", "standing person"
        Film landscape (3:2): "35mm", "film", "photo", "DSLR"
        Ultra-wide (21:9): "ultrawide", "cinema", "anamorphic"
        Panoramic (2:1): "panoramic", "wide panorama"
        Match input (match_input_image): "match aspect ratio", "same aspect ratio", "keep aspect ratio"

        CONTENT-BASED DETECTION:
        If no explicit aspect ratio is mentioned, choose based on typical content:
        - Input image + no specific aspect ratio → "match_input_image" 
        - Landscapes/scenery/cityscapes → "16:9"
        - People portraits/headshots → "3:4" 
        - Mobile/phone content → "9:16"
        - Traditional photos → "3:2" or "4:3"
        - Square objects/logos → "1:1"
        - Ultra-wide scenes → "21:9"
        - Panoramic views → "2:1"

        OTHER PARAMETERS:
        - Extract the main visual description for the prompt field (remove URLs)
        - Set input_image to the detected URL/path if found
        - Look for seed numbers, format preferences (PNG/JPG), safety requirements
        - Default output_format is "png"
        - Default safety_tolerance is 2

        EXAMPLES:
        "anime girl playing guitar, 16:9" → aspect_ratio: "16:9"
        "Make this a 90s cartoon, match aspect ratio" → aspect_ratio: "match_input_image" (if input image exists)
        "transform this image into oil painting style" → aspect_ratio: "match_input_image" (if input image exists)
        "square logo design" → aspect_ratio: "1:1"  
        "portrait of a warrior" → aspect_ratio: "3:4"
        """,
        output_type=ImageGenParams
    )

    result = await agent.run(prompt)
    params = result.output

    # Ensure input_image is set correctly
    if detected_input_image:
        params.input_image = detected_input_image

    return params


def generate_images_with_enhanced_save(params: ImageGenParams, prompt: str, custom_filename: str = None) -> Dict[
    str, str]:
    """
    Generate images with enhanced file management and Replicate ID extraction

    :param params: ImageGenParams object
    :param prompt: The prompt used for generation
    :param custom_filename: Optional custom filename (without extension)
    :return: Dictionary with file information
    """
    try:
        # Prepare input parameters
        input_params = params.model_dump()
        input_params["input_image"] = classify_path_or_url(input_params["input_image"])

        # Remove None values
        input_params = {k: v for k, v in input_params.items() if v is not None}

        # Generate image using Replicate
        print(f"🔄 Calling Replicate API with parameters: {input_params}")
        output = replicate.run("black-forest-labs/flux-kontext-pro", input=input_params)

        # Replicate returns a URL to the generated image
        output_url = str(output)
        print(f"📡 Replicate returned URL: {output_url}")

        # Download the image from the URL
        response = requests.get(output_url)
        response.raise_for_status()

        # Save with enhanced metadata using Replicate prediction ID
        result = save_image_with_metadata(
            image_data=response.content,
            original_url=output_url,
            prompt=prompt,
            params=params,
            custom_name=custom_filename
        )

        print(f"💾 Saved image: {result['filepath']}")
        print(f"🆔 Prediction ID: {result['prediction_id']}")

        return result

    except Exception as e:
        raise Exception(f"Error in enhanced image generation: {str(e)}")


def generate_images_with_custom_name(params: ImageGenParams, filename: str) -> str:
    """
    Generate images with custom filename (legacy compatibility)

    :param params: ImageGenParams object
    :param filename: Custom filename (without extension)
    :return: filepath of saved generated image
    """
    try:
        result = generate_images_with_enhanced_save(params, params.prompt, filename)
        return result["filepath"]

    except Exception as e:
        print(f"Error generating image with custom name: {str(e)}")
        return f"Error: {str(e)}"


def generate_images(params: ImageGenParams) -> str:
    """
    Generate images (legacy compatibility)

    :param params: ImageGenParams object with generation parameters
    :return: filepath of saved generated image
    """
    try:
        result = generate_images_with_enhanced_save(params, params.prompt)
        return result["filepath"]

    except Exception as e:
        print(f"Error generating image: {str(e)}")
        return f"Error: {str(e)}"


def classify_path_or_url(param):
    """
    Determine if parameter is a local filepath, URL, or None
    """
    if param is None or param == "":
        return None

    param = str(param).strip()

    # Check if it's a URL
    try:
        parsed = urlparse(param)
        if parsed.scheme and parsed.netloc:
            return param
    except:
        pass

    # Check if it's a local filepath
    try:
        if (os.path.isabs(param) or
                param.startswith('./') or param.startswith('../') or
                '/' in param or '\\' in param or
                os.path.exists(param)):
            return open(param, "rb")
    except:
        pass

    # Try to open as file
    try:
        if param:
            return open(param, "rb")
    except:
        pass

    return None


def get_saved_images_info() -> Dict[str, Any]:
    """
    Get information about all saved images in the output directory
    """
    if not OUTPUT_DIR.exists():
        return {"total_images": 0, "images": []}

    images = []
    image_files = list(OUTPUT_DIR.glob("*.png")) + list(OUTPUT_DIR.glob("*.jpg")) + list(
        OUTPUT_DIR.glob("*.jpeg")) + list(OUTPUT_DIR.glob("*.webp"))

    for image_file in image_files:
        # Look for corresponding metadata file
        metadata_file = OUTPUT_DIR / f"{image_file.stem}_metadata.json"

        image_info = {
            "filename": image_file.name,
            "filepath": str(image_file),
            "size_bytes": image_file.stat().st_size if image_file.exists() else 0,
            "created": datetime.fromtimestamp(image_file.stat().st_ctime).isoformat() if image_file.exists() else None
        }

        # Load metadata if available
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    image_info.update(metadata)
            except:
                pass

        images.append(image_info)

    # Sort by creation time, newest first
    images.sort(key=lambda x: x.get("created", ""), reverse=True)

    return {
        "total_images": len(images),
        "output_directory": str(OUTPUT_DIR),
        "images": images
    }


# Test function for the enhanced system
async def test_enhanced_image_generation():
    """
    Test the enhanced image generation with file management
    """
    print("🧪 Testing Enhanced Image Generation System")
    print("=" * 60)

    test_prompt = "anime girl playing guitar in ancient Feudal Japan, aspect ratio is 16:9"

    try:
        print(f"📝 Test prompt: '{test_prompt}'")

        # Extract parameters
        print("🔍 Extracting parameters...")
        params = await formatting_params_from_prompt(test_prompt)
        print(f"✅ Parameters extracted:")
        print(f"   📐 Aspect ratio: {params.aspect_ratio}")
        print(f"   🎨 Output format: {params.output_format}")
        print(f"   📝 Cleaned prompt: {params.prompt}")

        # Generate image with enhanced save
        print("🖼️ Generating image with enhanced file management...")
        result = generate_images_with_enhanced_save(params, test_prompt)

        print(f"✅ Generation successful!")
        print(f"   📁 File: {result['filepath']}")
        print(f"   🆔 Prediction ID: {result['prediction_id']}")
        print(f"   🔗 Original URL: {result['original_url']}")
        print(f"   📋 Metadata: {result['metadata_file']}")

        return True

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False


if __name__ == "__main__":
    import asyncio

    print("🚀 Testing Enhanced AI Tools with File Management")
    asyncio.run(test_enhanced_image_generation())