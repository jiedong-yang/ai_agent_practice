from mcp.server.fastmcp import FastMCP
from tools import get_host_info
from ai_tools import formatting_params_from_prompt, generate_images_with_custom_name, ImageGenParams
from pydantic_ai import Agent
from typing import List, Dict, Any, Optional
import json
import os
import re
from datetime import datetime
from pathlib import Path

# Create output directory
OUTPUT_DIR = Path("generated_images")
OUTPUT_DIR.mkdir(exist_ok=True)

mcp = FastMCP("Enhanced Image Generation MCP")
mcp.add_tool(get_host_info)

# Specialized agents for different tasks
prompt_improvement_agent = Agent(
    'google-gla:gemini-2.5-flash',
    system_prompt="""
    You are an expert prompt engineer specializing in image generation. Your job is to take a basic prompt 
    and transform it into a detailed, high-quality prompt that will produce stunning images.

    PROMPT IMPROVEMENT GUIDELINES:
    1. Expand basic descriptions into vivid, detailed scenes
    2. Add specific artistic styles (anime, realistic, oil painting, etc.)
    3. Include lighting descriptions (soft, dramatic, golden hour, etc.)
    4. Add composition details (close-up, wide shot, low angle, etc.)
    5. Specify image quality terms (high resolution, detailed, masterpiece, etc.)
    6. Include atmospheric elements (mood, weather, ambiance)
    7. Add technical photography terms when appropriate
    8. Maintain the core concept while enhancing all aspects

    TEXT AND UI ELEMENTS:
    - Text in images (buttons, logos, signs, UI) is NORMAL and EXPECTED
    - Include specific text formatting: "bold white text", "glowing letters", "stylized font"
    - Describe text placement: "bottom left corner", "center", "floating above"
    - Enhance text visibility: "outlined text", "text with drop shadow", "contrasting background"

    IMAGE-TO-IMAGE TRANSFORMATIONS:
    - Input images (URLs, file paths) enable style transfer, editing, and variations
    - For style transformations: "convert to 90s cartoon style", "make it anime art", "oil painting effect"
    - For editing: "remove text", "change background", "modify colors", "add elements"
    - Enhance transformation descriptions: "vibrant 90s cartoon aesthetic with bold outlines and bright colors"
    - Maintain core elements while applying new style: "preserve character pose and composition"

    GAMING/ADVERTISING CONTENT:
    - For game ads: emphasize dynamic action, vibrant colors, eye-catching design
    - Include marketing elements: "call-to-action button", "promotional design", "engaging layout"
    - Enhance visual impact: "explosive effects", "dramatic poses", "compelling composition"

    STRUCTURE YOUR IMPROVEMENTS:
    - Main subject with enhanced details
    - Setting/environment with rich descriptions  
    - Text elements with clear styling and placement
    - Lighting and mood
    - Art style and technical specifications
    - Composition and framing

    EXAMPLE:
    Basic: "gaming ad with anime girl, sword, snow, PLAY NOW button"
    Improved: "Professional mobile game advertisement featuring a fierce anime warrior girl with flowing silver hair, wielding an ornate katana with blue energy effects. She stands confidently in a snow-covered feudal Japanese village with traditional architecture. Bold white 'PLAY NOW' text with glowing blue outline positioned in bottom left corner on dark semi-transparent background. Dynamic action pose, vibrant colors, high contrast lighting, cinematic composition, detailed anime art style, mobile game UI aesthetic."

    IMG2IMG EXAMPLE:
    Basic: "Make this a 90s cartoon"
    Improved: "Transform into vibrant 90s cartoon style with bold black outlines, bright saturated colors, and classic animation aesthetic. Maintain original composition and character pose while applying retro cartoon visual effects, cell-shaded appearance, and nostalgic animation quality reminiscent of 1990s animated series."

    Return ONLY the improved prompt text, no explanations or additional commentary.
    """
)

prompt_generator_agent = Agent(
    'google-gla:gemini-2.5-flash',
    system_prompt="""
    You are a creative prompt generator for image generation. When asked to generate multiple prompts,
    create diverse, detailed, and visually interesting prompts that would result in varied and 
    appealing images. Each prompt should be descriptive and suitable for AI image generation.

    Return the prompts as a JSON list of strings.
    """
)


def extract_replicate_id_from_url(url: str) -> str:
    """
    Extract the prediction ID from a Replicate URL for consistent naming

    Example: https://replicate.delivery/pbxt/abc123.png -> abc123
    """
    try:
        # Extract filename from URL
        filename = url.split('/')[-1]
        # Remove extension
        prediction_id = filename.split('.')[0]
        return prediction_id
    except:
        # Fallback to timestamp if extraction fails
        return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_image_with_metadata(image_data: bytes, original_url: str, prompt: str, params: ImageGenParams) -> str:
    """
    Save image with consistent naming and metadata
    """
    # Extract prediction ID from Replicate URL
    prediction_id = extract_replicate_id_from_url(original_url)

    # Create filename with prediction ID
    filename = f"{prediction_id}.{params.output_format}"
    filepath = OUTPUT_DIR / filename

    # Save image
    with open(filepath, 'wb') as f:
        f.write(image_data)

    # Save metadata
    metadata = {
        "filename": filename,
        "original_url": original_url,
        "prompt": prompt,
        "parameters": params.model_dump(),
        "generated_at": datetime.now().isoformat(),
        "prediction_id": prediction_id
    }

    metadata_filepath = OUTPUT_DIR / f"{prediction_id}_metadata.json"
    with open(metadata_filepath, 'w') as f:
        json.dump(metadata, f, indent=2)

    return str(filepath)


@mcp.tool()
async def improve_prompt(basic_prompt: str, style_hints: str = "") -> str:
    """
    Improve a basic prompt into a detailed, high-quality image generation prompt.

    :param basic_prompt: The basic prompt to improve
    :param style_hints: Optional style hints (e.g., "cinematic", "anime", "realistic")
    :return: Improved detailed prompt
    """
    try:
        enhancement_request = f"Improve this image generation prompt: '{basic_prompt}'"
        if style_hints:
            enhancement_request += f" with {style_hints} style elements"

        result = await prompt_improvement_agent.run(enhancement_request)
        return result.output.strip()

    except Exception as e:
        return f"Error improving prompt: {str(e)}"


@mcp.tool()
async def generate_multiple_improved_prompts(basic_prompt: str, count: int = 3, style_variety: str = "mixed") -> List[
    str]:
    """
    Generate multiple improved variations of a basic prompt.

    :param basic_prompt: The basic prompt to improve
    :param count: Number of variations to generate
    :param style_variety: Style variety ("mixed", "anime", "realistic", "artistic")
    :return: List of improved prompt variations
    """
    try:
        request = f"""
        Create {count} different improved variations of this prompt: '{basic_prompt}'

        Style variety: {style_variety}

        Make each variation unique in:
        - Art style and technique
        - Lighting and mood  
        - Composition and framing
        - Environmental details
        - Character details and pose

        Return as a JSON array of strings.
        """

        result = await prompt_generator_agent.run(request)

        try:
            prompts = json.loads(result.output)
            return prompts[:count]
        except json.JSONDecodeError:
            # Fallback parsing
            lines = result.output.strip().split('\n')
            prompts = [line.strip('- ').strip().strip('"').strip("'") for line in lines if line.strip()]
            return [p for p in prompts if p and len(p) > 20][:count]

    except Exception as e:
        return [f"Error generating improved prompts: {str(e)}"]


@mcp.tool()
async def image_generator_with_improvement(prompt: str, should_improve_prompt: bool = False, style_hints: str = "") -> \
Dict[str, Any]:
    """
    Generate an image with optional prompt improvement.

    :param prompt: Original prompt
    :param should_improve_prompt: Whether to improve the prompt first
    :param style_hints: Optional style hints for improvement
    :return: Dictionary with generation results and metadata
    """
    try:
        # Step 1: Improve prompt if requested
        if should_improve_prompt:
            print("🎨 Improving prompt...")
            improved_prompt = await improve_prompt(prompt, style_hints)
            print(f"✨ Original: {prompt}")
            print(f"✨ Improved: {improved_prompt}")
            final_prompt = improved_prompt
        else:
            final_prompt = prompt

        # Step 2: Extract parameters
        print("📋 Extracting parameters...")
        image_params = await formatting_params_from_prompt(final_prompt)

        # Step 3: Generate image with enhanced save functionality
        print("🖼️ Generating image...")
        result = generate_images_with_enhanced_save(image_params, final_prompt)

        return {
            "success": True,
            "original_prompt": prompt,
            "final_prompt": final_prompt,
            "improved": should_improve_prompt,
            "filepath": result["filepath"],
            "prediction_id": result["prediction_id"],
            "parameters": image_params.model_dump()
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "original_prompt": prompt
        }


@mcp.tool()
async def batch_generator_with_improvements(
        basic_prompt: str,
        count: int = 3,
        should_improve_prompts: bool = True,
        style_variety: str = "mixed"
) -> Dict[str, Any]:
    """
    Generate multiple images with improved prompt variations.

    :param basic_prompt: The basic prompt to work from
    :param count: Number of images to generate
    :param should_improve_prompts: Whether to improve prompts
    :param style_variety: Style variety for improvements
    :return: Dictionary with all generation results
    """
    results = {
        "basic_prompt": basic_prompt,
        "successful": [],
        "failed": [],
        "total_requested": count
    }

    try:
        # Step 1: Generate improved prompts
        if should_improve_prompts:
            print(f"🎨 Generating {count} improved prompt variations...")
            prompts = await generate_multiple_improved_prompts(basic_prompt, count, style_variety)
        else:
            # Use variations of the basic prompt
            prompts = [basic_prompt] * count

        print(f"✅ Generated {len(prompts)} prompts")

        # Step 2: Generate images from each prompt
        for i, prompt in enumerate(prompts):
            try:
                print(f"🖼️ Generating image {i + 1}/{len(prompts)}...")

                # Extract parameters and generate
                image_params = await formatting_params_from_prompt(prompt)
                result = generate_images_with_enhanced_save(image_params, prompt)

                results["successful"].append({
                    "index": i + 1,
                    "prompt": prompt,
                    "filepath": result["filepath"],
                    "prediction_id": result["prediction_id"],
                    "parameters": image_params.model_dump()
                })

                print(f"✅ Generated image {i + 1}: {result['filepath']}")

            except Exception as e:
                error_msg = f"Error generating image {i + 1}: {str(e)}"
                print(f"❌ {error_msg}")

                results["failed"].append({
                    "index": i + 1,
                    "prompt": prompt,
                    "error": str(e)
                })

        # Summary
        successful_count = len(results["successful"])
        success_rate = (successful_count / count) * 100 if count > 0 else 0
        results["summary"] = f"Generated {successful_count}/{count} images ({success_rate:.1f}% success rate)"

        return results

    except Exception as e:
        results["failed"].append({
            "error": f"Batch generation failed: {str(e)}"
        })
        return results


def generate_images_with_enhanced_save(params: ImageGenParams, prompt: str) -> Dict[str, str]:
    """
    Enhanced image generation with proper file management
    """
    import replicate
    from ai_tools import classify_path_or_url

    try:
        # Prepare input parameters
        input_params = params.model_dump()
        input_params["input_image"] = classify_path_or_url(input_params["input_image"])

        # Remove None values
        input_params = {k: v for k, v in input_params.items() if v is not None}

        # Generate image
        output = replicate.run("black-forest-labs/flux-kontext-pro", input=input_params)

        # Get the output URL (Replicate returns a URL)
        output_url = str(output)

        # Download the image
        import requests
        response = requests.get(output_url)
        response.raise_for_status()

        # Save with enhanced metadata
        filepath = save_image_with_metadata(response.content, output_url, prompt, params)
        prediction_id = extract_replicate_id_from_url(output_url)

        return {
            "filepath": filepath,
            "prediction_id": prediction_id,
            "original_url": output_url
        }

    except Exception as e:
        raise Exception(f"Error in enhanced image generation: {str(e)}")


# Keep original tools for backward compatibility
@mcp.tool()
async def image_generator(prompt: str) -> str:
    """Original image generator for backward compatibility"""
    result = await image_generator_with_improvement(prompt, should_improve_prompt=False)
    if result["success"]:
        return result["filepath"]
    else:
        return f"Error: {result['error']}"


@mcp.tool()
async def generate_creative_prompts(count: int, theme: str = "general") -> List[str]:
    """Generate creative prompts (original functionality)"""
    try:
        user_request = f"Generate {count} diverse and creative image generation prompts"
        if theme != "general":
            user_request += f" with a {theme} theme"
        user_request += ". Return as JSON array."

        result = await prompt_generator_agent.run(user_request)

        try:
            prompts = json.loads(result.output)
            return prompts[:count]
        except json.JSONDecodeError:
            lines = result.output.strip().split('\n')
            prompts = [line.strip('- ').strip().strip('"').strip("'") for line in lines if line.strip()]
            return [p for p in prompts if p and len(p) > 10][:count]

    except Exception as e:
        fallback_prompts = [
            f"A beautiful {theme} scene with rich details and vibrant colors",
            f"An artistic {theme} composition with dramatic lighting",
            f"A stunning {theme} landscape with intricate textures"
        ]
        return fallback_prompts[:count]


def main() -> None:
    print("Enhanced MCP Server with Prompt Improvement starting...")
    print(f"Images will be saved to: {OUTPUT_DIR.absolute()}")
    mcp.run(transport="streamable-http")


if __name__ == '__main__':
    main()