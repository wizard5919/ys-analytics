from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import os

# Create a blank image with dark blue background
width, height = 800, 800
img = Image.new('RGB', (width, height), color='#0A1F44')
draw = ImageDraw.Draw(img)

# Draw a simple teal brain icon (neural network: central node with branches)
teal = '#00C2FF'
# Central node
draw.ellipse((300, 200, 500, 400), fill=teal, outline=teal)
# Branching nodes and connections
nodes = [(200, 150), (600, 150), (200, 450), (600, 450), (400, 50), (400, 550)]
for node in nodes:
    draw.line((400, 300) + node, fill=teal, width=5)
    draw.ellipse((node[0]-20, node[1]-20, node[0]+20, node[1]+20), fill=teal, outline=teal)

# Add text: "YS Analytics" in white
try:
    font_large = ImageFont.truetype("arialbd.ttf", 100)  # Bold Arial; adjust if font not found
except:
    font_large = ImageFont.load_default()  # Fallback
draw.text((150, 400), "YS Analytics", fill='white', font=font_large)

# Add subtitle: "AI FOR FINANCE" in teal
try:
    font_small = ImageFont.truetype("arial.ttf", 50)
except:
    font_small = ImageFont.load_default()
draw.text((250, 550), "AI FOR FINANCE", fill=teal, font=font_small)

# Save the image to assets/logo.png
assets_path = 'assets/logo.png'
img.save(assets_path)
print(f"Logo saved to {assets_path}")
