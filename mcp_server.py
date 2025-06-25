from mcp.server.fastmcp import FastMCP
from tools import get_host_info
from ai_tools import formatting_params_from_prompt, generate_images

mcp = FastMCP("Host info mcp")
mcp.add_tool(get_host_info)
mcp.add_tool(formatting_params_from_prompt)

@mcp.tool()
def image_generator(prompt) -> str:
    """ generate images based on prompt, formatting the prompt to extract parameters from the prompt first,
    then use the parameters to generate images by inferencing with replicate.
    Image generation model used is Flux Kontext.
    :param prompt:
    :return: saved image file path
    """
    image_params = formatting_params_from_prompt(prompt)
    # print(image_params)

    return generate_images(image_params)


def main() -> None:
    print("MCP Server starting...")
    mcp.run(transport="streamable-http")

if __name__ == '__main__':
    main()