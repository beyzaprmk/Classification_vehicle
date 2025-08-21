from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

# Modeli yükle
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5"
)
pipe = pipe.to("cpu")  # CPU kullanımı

# Başlangıç resmi
init_image = Image.open(r"path").convert("RGB").resize((512, 512))

# Prompt ile yeni varyasyon üret
prompt = "Photograph of a green and white Dywatrade Auto Service mobile van parked on a sunny urban street. The van features a large blue swoosh on the side, with white text reading Mobile Service Van and a logo. Contact details, including phone numbers, are displayed below. The background includes beige buildings with arched windows, a clear blue sky, and a few parked cars. The van's side mirrors and front bumper are visible, and the ground is paved with red and gray bricks. The van's design is modern and professional, with clean lines and a polished appearance."

images = pipe(prompt=prompt, image=init_image, strength=0.6).images

# Sonuçları göster ve kaydet
images[0].show()
images[0].save("./car_variation2.png")
