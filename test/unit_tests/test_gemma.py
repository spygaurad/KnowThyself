from langchain_community.llms import ollama
import base64
from io import BytesIO

from IPython.display import HTML, display
from PIL import Image


def convert_to_base64(pil_image):
    """
    Convert PIL images to Base64 encoded strings

    :param pil_image: PIL image
    :return: Re-sized Base64 string
    """
    if pil_image.mode == "RGBA":
        pil_image = pil_image.convert("RGB")

    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")  # You can change the format if needed
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def plt_img_base64(img_base64):
    """
    Display base64 encoded string as image

    :param img_base64:  Base64 string
    """
    # Create an HTML img tag with the base64 string as the source
    image_html = f'<img src="data:image/jpeg;base64,{img_base64}" />'
    # Display the image by rendering the HTML
    display(HTML(image_html))


file_path = "/home/prasais/projects/KnowYourLLM/attention_output.png"
pil_image = Image.open(file_path)
image_b64 = convert_to_base64(pil_image)

llm = ollama.Ollama(
                        # base_url=ollama_base_url, 
                        # model='mapler/gpt2',
                        model = 'gemma3:27b'
                        )

llm_with_image_context = llm.bind(images=[image_b64])
response = llm_with_image_context.invoke("Explain the attention pattern in the image")
print(response)