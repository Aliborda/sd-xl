import torch
from diffusers import AutoPipelineForText2Image

print("Cargando Modelo...")

modelo = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float32
        # FloatXX puede ser 32 o 16
        # (indica la cantidad de decimales de precisión con el que el modelo creará la imagen)
        # Si mi pc genera una imagen negra, cambiar a 16
 )
 
modelo = modelo.to("cpu")
# Le decimos que se ejecute con la CPU y RAM
# # Se puede cambiar a cuda - con NVIDIA
# # mps - Mac con procesador MX
# # xpu - Grfica Intel iris

prompt = input("Escribe el prompt de la imagen: ")
negrative_prompt = "lowres, bad anatomy, bad hands, text, watermark, signature, blurry, worst quality, low quality, jpeg artifacts, deformed, extra fingers, mutated hands, poorly drawn face, out of frame, cropped"

imagen = modelo(
        prompt=prompt,
        negrative_prompt=negrative_prompt,
        num_inference_steps=30,
        guidance_scale=7.0,
        height=1024,
        width=1024
).images[0]

imagen.save("imagen.png")
print("Imagen Guardada")

