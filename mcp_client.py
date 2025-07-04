import asyncio
import json
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStreamableHTTP
from typing import List, Any

load_dotenv()

server = MCPServerStreamableHTTP('http://localhost:8000/mcp')

# Enhanced agent with prompt improvement understanding
agent = Agent(
    'google-gla:gemini-2.5-flash',
    mcp_servers=[server],
    system_prompt="""
    You are an expert AI assistant with advanced image generation capabilities. You have access to:

    CORE TOOLS:
    1. **image_generator_with_improvement(prompt, should_improve_prompt, style_hints)** - Generate single images with optional prompt enhancement
    2. **batch_generator_with_aspect_preservation(basic_prompt, count, should_improve_prompts, style_variety)** - Generate multiple images with improved prompts AND preserved aspect ratio
    3. **improve_prompt(basic_prompt, style_hints)** - Just improve a prompt without generating
    4. **generate_multiple_improved_prompts(basic_prompt, count, style_variety)** - Create multiple improved variations

    IMPORTANT IMAGE GENERATION CAPABILITIES:
    - The Flux Kontext model can generate ANY visual content described in text prompts
    - This includes text, logos, buttons, UI elements, signs, banners, etc. as part of the image
    - Text generation (like "PLAY NOW", "START", logos, etc.) is a STANDARD capability
    - **IMG2IMG TRANSFORMATIONS**: Can use input images (URLs or file paths) for style transfer, editing, variations
    - **Image editing capabilities**: Remove text, style transfer, object modification, background changes
    - **Input image support**: URLs (like replicate.delivery links) and local file paths work perfectly
    - Never refuse text generation OR image-to-image requests - they are normal capabilities
    - Always proceed confidently with any visual prompt including text elements AND input images

    TEXT IN IMAGES:
    When users request text in images (like "PLAY NOW button", "game UI", "advertisement text"):
    - This is NORMAL image generation, not overlay/editing
    - Include the text description directly in the image prompt
    - Improve prompts to make text more visible and well-designed
    - Never suggest external editing tools for basic text generation

    IMAGE-TO-IMAGE WORKFLOWS:
    When users provide image URLs or file paths with transformation requests:
    - Extract the input image URL/path and include in the image generation parameters
    - Use "match_input_image" aspect ratio when requested to match input dimensions
    - Common transformations: style transfer, cartoon conversion, artistic filters, editing
    - Example: "using this image [URL] make it 90s cartoon style" → Use input_image parameter
    - Never refuse img2img requests - this is a core Kontext model capability

    LEGACY TOOLS (for backward compatibility):
    - image_generator(prompt) - Basic single image generation
    - generate_creative_prompts(count, theme) - Basic prompt generation

    PROMPT IMPROVEMENT CAPABILITIES:
    When users ask to "improve", "enhance", "make better", or "help me improve" their prompts:
    - Use improve_prompt for single prompt enhancement
    - Use generate_multiple_improved_prompts for multiple variations
    - Use image_generator_with_improvement with improve_prompt=True for generation with improvement

    STYLE VARIETIES AVAILABLE:
    - "mixed" - Various artistic styles
    - "anime" - Anime/manga focused
    - "realistic" - Photorealistic styles  
    - "artistic" - Painterly/artistic styles
    - "cinematic" - Movie/film styles

    WORKFLOW DETECTION:
    1. **Prompt improvement only**: "improve this prompt: ..." → Use improve_prompt
    2. **Generate with improvement**: "generate ... help me improve" → Use image_generator_with_improvement with should_improve_prompt=True
    3. **Multiple improved variations**: "generate 3 improved versions of ..." → Use batch_generator_with_aspect_preservation
    4. **Basic generation**: "generate an image of ..." → Use image_generator_with_improvement with should_improve_prompt=False
    5. **Image-to-image**: "using this image [URL/path] ..." → Use image_generator_with_improvement with input image
    6. **Style transfer**: "make this [URL] into 90s cartoon" → Extract URL, use img2img workflow

    CRITICAL: For batch generation, ALWAYS use batch_generator_with_aspect_preservation to maintain 
    aspect ratios from the original prompt (e.g., "widescreen" → 16:9, "portrait" → 3:4)

    IMAGE INPUT DETECTION:
    - Look for URLs (especially replicate.delivery, https://, http://)
    - Look for file paths (local files, relative paths)
    - Keywords: "using this image", "with this photo", "from this picture", "transform this"
    - When found: extract the image reference and use as input_image parameter

    FILE MANAGEMENT:
    - All images are automatically saved to "generated_images/" folder
    - Filenames use Replicate prediction IDs for consistency
    - Metadata is saved alongside each image
    - Always mention the saved file location to users

    IMPORTANT BEHAVIOR:
    - Always be confident about parameter extraction (aspect ratios, etc.)
    - NEVER refuse text generation in images - it's a standard capability
    - NEVER refuse image-to-image transformations - Kontext model excels at this
    - When users request text/buttons/UI elements, treat as normal image generation
    - When users provide image URLs/paths, extract them and use img2img workflow
    - When improvement is requested, show both original and improved prompts
    - Provide clear information about saved files and their locations
    - Offer to generate variations or improvements when appropriate
    - Process ALL visual requests confidently: text, logos, UI elements, img2img, style transfer, etc.
    - For "match aspect ratio" requests with input images, use "match_input_image" aspect ratio
    """
)


async def detect_request_type(user_input: str) -> dict:
    """
    Enhanced request detection for prompt improvement workflows
    """
    user_lower = user_input.lower()

    # Check for improvement keywords
    improvement_keywords = ["improve", "enhance", "better", "help me improve", "make better", "optimize"]
    has_improvement = any(keyword in user_lower for keyword in improvement_keywords)

    # Check for generation keywords
    generation_keywords = ["generate", "create", "make"]
    has_generation = any(keyword in user_lower for keyword in generation_keywords)

    # Check for numbers
    import re
    numbers = re.findall(r'\b(\d+)\b', user_input)
    is_batch = numbers and int(numbers[0]) > 1

    # Determine workflow type
    if has_improvement and not has_generation:
        return {"type": "prompt_improvement_only", "batch": is_batch}
    elif has_improvement and has_generation:
        return {"type": "generate_with_improvement", "batch": is_batch}
    elif is_batch:
        return {"type": "batch_generation", "batch": True}
    else:
        return {"type": "single_generation", "batch": False}


async def main() -> None:
    history: List[Any] = []

    print("🎨 Enhanced AI Image Generation Assistant")
    print("✨ Now with intelligent prompt improvement!")
    print("\n🔧 New capabilities:")
    print("📝 Prompt improvement: 'improve this prompt: anime girl with sword'")
    print("🎭 Generate with improvement: 'generate a landscape, help me improve the prompt'")
    print("🚀 Multiple improved versions: 'generate 3 improved versions of anime girl'")
    print("💾 Smart file management: All images saved with Replicate IDs")
    print("\n💡 Style varieties: mixed, anime, realistic, artistic, cinematic")
    print("-" * 70)

    async with agent.run_mcp_servers():
        while True:
            try:
                user_prompt = input("\n💬 Enter your message (or 'quit' to exit): ")

                if user_prompt.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break

                if not user_prompt.strip():
                    continue

                print("🤔 Processing your request...")

                # Enhanced request analysis
                request_info = await detect_request_type(user_prompt)

                request_icons = {
                    "prompt_improvement_only": "📝",
                    "generate_with_improvement": "🎨",
                    "batch_generation": "🎭",
                    "single_generation": "🖼️"
                }

                request_descriptions = {
                    "prompt_improvement_only": "Prompt improvement request",
                    "generate_with_improvement": "Generation with prompt improvement",
                    "batch_generation": "Batch generation request",
                    "single_generation": "Single image generation"
                }

                icon = request_icons.get(request_info["type"], "🎯")
                description = request_descriptions.get(request_info["type"], "Processing request")

                print(f"{icon} Detected: {description}")

                if request_info["type"] == "generate_with_improvement":
                    print("✨ Will improve your prompt before generation")
                elif request_info["batch"]:
                    print(f"🔄 Will generate multiple images")

                # Run the agent
                result = await agent.run(user_prompt, message_history=history)
                history = list(result.all_messages())

                print(f"\n🤖 Assistant: {result.output}")

                # Enhanced file detection and display
                output_text = result.output
                if "generated_images/" in output_text or any(
                        ext in output_text.lower() for ext in ['.png', '.jpg', '.jpeg', '.webp']):
                    print("\n📁 Generated files saved to 'generated_images/' folder:")

                    # Extract file paths
                    import re
                    file_patterns = [
                        r'generated_images/[\w\-_]+\.(png|jpg|jpeg|webp)',
                        r'[\w\-_]+\.(png|jpg|jpeg|webp)'
                    ]

                    found_files = set()
                    for pattern in file_patterns:
                        matches = re.findall(pattern, output_text, re.IGNORECASE)
                        for match in matches:
                            if isinstance(match, tuple):
                                filename = match[0] if len(match) > 1 else match
                            else:
                                filename = match
                            found_files.add(filename)

                    # Also look for explicit mentions
                    lines = output_text.split('\n')
                    for line in lines:
                        if any(ext in line.lower() for ext in ['.png', '.jpg', '.jpeg', '.webp']):
                            # Try to extract filename from line
                            import os
                            for word in line.split():
                                if any(ext in word.lower() for ext in ['.png', '.jpg', '.jpeg', '.webp']):
                                    found_files.add(os.path.basename(word.strip('.,!?')))

                    for filename in sorted(found_files):
                        print(f"   📸 {filename}")

                # Offer follow-up suggestions
                if request_info["type"] == "single_generation" and "improve" not in user_prompt.lower():
                    print(
                        "\n💡 Want to improve this prompt? Just ask 'improve that prompt' or 'generate an improved version'")
                elif request_info["type"] == "prompt_improvement_only":
                    print("\n💡 Want to generate an image with this improved prompt? Just ask 'generate that image'")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                print("Please try again with a different request.")


if __name__ == '__main__':
    asyncio.run(main())