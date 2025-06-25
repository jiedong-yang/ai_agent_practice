import replicate
import os

from dotenv import load_dotenv
from urllib.parse import urlparse
from typing import List, Any, Dict, Optional
from pydantic_ai import Agent
from pydantic import BaseModel, Field

load_dotenv()

class ImageGenParams(BaseModel):
    prompt: str = Field(description="Image prompt for image generation model")
    input_image: Optional[str] = Field(default=None, description="Path or url to input image for img2img or None")
    aspect_ratio: str = Field(default="1:1", description="Image aspect ratio like 16:9, 1:1, 4:3")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    output_format: str = Field(default="png", description="Output image format")
    safety_tolerance: int = Field(default=2, ge=0, le=6, description="Safety filter level 0-6")


async def formatting_params_from_prompt(prompt: str) -> ImageGenParams:
    agent = Agent('google-gla:gemini-2.5-flash',
                  system_prompt="""
                  extract image generation prompt from input
                  """, output_type=ImageGenParams)
    result = await agent.run(prompt)
    # print(result)
    return result.output


def generate_images(params: ImageGenParams) -> str:
    """
    generating images from a prompt and numerous parameters,
    by using Flux Kontext model hosted on replicate
    :param params: ImageGenParams
    :return: filepath of saved generated image
    """

    input = params.model_dump()

    input["input_image"] = classify_path_or_url(input["input_image"])

    # input = {
    #     "prompt": prompt,
    #     "input_image": image,
    #     "aspect_ratio": aspect_ratio,
    #     "seed": seed,
    #     "output_format": output_format,
    #     "safety_tolerance": safety_tolerance
    # }

    output = replicate.run("black-forest-labs/flux-kontext-pro", intput=input)

    output_filepath = "output.png"
    with open(output_filepath, "wb") as file:
        file.write(output.read())
    return output_filepath


def classify_path_or_url(param):
    """
    Determine if parameter is a local filepath, URL, or None

    Args:
        param: The parameter to classify

    Returns:
        None if param is None,
        param if param is a url,
        open(param, "rb") if param is a local filepath
    """

    # Check if parameter is None or empty
    if param is None or param == "":
        return None

    # Convert to string if not already
    param = str(param).strip()

    # Check if it's a URL
    try:
        parsed = urlparse(param)
        # Valid URL should have scheme (http, https, ftp, etc.)
        if parsed.scheme and parsed.netloc:
            return param
    except:
        pass

    # Check if it's a local filepath
    # This checks for common path patterns
    if (os.path.isabs(param) or  # Absolute path
            param.startswith('./') or param.startswith('../') or  # Relative paths
            '/' in param or '\\' in param or  # Contains path separators
            os.path.exists(param)):  # File/directory actually exists
        return open(param, "rb")

    # If none of the above, treat as filepath (could be a filename) and return opened local file
    return open(param, "rb")

